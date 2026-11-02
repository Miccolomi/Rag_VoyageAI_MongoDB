"""Cifratura lato client (CSFLE esplicita) della prima domanda di ogni sessione.

L'applicazione cifra il valore prima di inviarlo ad Atlas e lo decifra dopo averlo letto:
nel database, nei backup e per chi amministra il cluster il campo è solo un dato binario.

- Chiave master: 96 byte in un file locale fuori dal progetto (config.CSFLE_MASTER_KEY_PATH).
  In produzione starebbe in un servizio di gestione chiavi (KMS).
- Chiave dati: un documento nel key vault (config.CSFLE_KEY_VAULT_NAMESPACE), cifrato con la
  chiave master e riconosciuto per nome (config.CSFLE_KEY_ALT_NAME).
- Algoritmo casuale: lo stesso testo dà ogni volta un cifrato diverso. Sul campo non si fanno
  ricerche, quindi è la scelta più protetta.

Non si usa la cifratura automatica del client: la documentazione indica che `$rerank` non è
supportato dai client configurati per la cifratura automatica, e la ricerca deve restare intatta.

Preparazione (una volta):  .venv/bin/python -m src.field_encryption setup
"""
from __future__ import annotations

import base64
import os
import sys
import threading
from functools import lru_cache
from typing import Any

from bson.binary import Binary, UuidRepresentation
from bson.codec_options import CodecOptions

from . import config

ALGORITHM_NAME = "AEAD_AES_256_CBC_HMAC_SHA_512-Random"
ENCRYPTED_SUBTYPE = 6  # sottotipo BSON dei valori cifrati lato client
ENCRYPTED_FIELD = "first_question"
_KEY_READY = threading.Event()


class EncryptionUnavailable(RuntimeError):
    """La cifratura non è utilizzabile: chiave master assente o libreria mancante."""


def is_configured() -> bool:
    return config.CSFLE_MASTER_KEY_PATH.is_file()


def _master_key() -> bytes:
    path = config.CSFLE_MASTER_KEY_PATH
    if not path.is_file():
        raise EncryptionUnavailable("Chiave master locale non trovata: esegui la preparazione della cifratura.")
    key = path.read_bytes()
    if len(key) != 96:
        raise EncryptionUnavailable("La chiave master locale deve essere di 96 byte.")
    return key


@lru_cache(maxsize=1)
def client_encryption():
    try:
        from pymongo.encryption import ClientEncryption
    except ImportError as exc:  # pymongocrypt assente
        raise EncryptionUnavailable("Libreria di cifratura non installata (pymongocrypt).") from exc
    return ClientEncryption(
        {"local": {"key": _master_key()}},
        config.CSFLE_KEY_VAULT_NAMESPACE,
        config.get_client(),
        CodecOptions(uuid_representation=UuidRepresentation.STANDARD),
    )


def _key_vault():
    db_name, coll_name = config.CSFLE_KEY_VAULT_NAMESPACE.split(".", 1)
    return config.get_client()[db_name][coll_name]


def ensure_data_key() -> Binary:
    """Crea (se manca) la chiave dati e l'indice unico su keyAltNames richiesto dal key vault."""
    _key_vault().create_index(
        "keyAltNames", unique=True,
        partialFilterExpression={"keyAltNames": {"$exists": True}})
    ce = client_encryption()
    existing = ce.get_key_by_alt_name(config.CSFLE_KEY_ALT_NAME)
    key_id = existing["_id"] if existing else ce.create_data_key(
        "local", key_alt_names=[config.CSFLE_KEY_ALT_NAME])
    _KEY_READY.set()
    return key_id


def is_encrypted(value: Any) -> bool:
    return isinstance(value, Binary) and value.subtype == ENCRYPTED_SUBTYPE


def encrypt_text(value: str) -> Any:
    """Cifra se la chiave è disponibile; altrimenti restituisce il testo (la demo non si ferma)."""
    if not is_configured():
        return value
    from pymongo.encryption import Algorithm
    if not _KEY_READY.is_set():
        ensure_data_key()
    return client_encryption().encrypt(
        value, Algorithm.AEAD_AES_256_CBC_HMAC_SHA_512_Random, key_alt_name=config.CSFLE_KEY_ALT_NAME)


def decrypt_text(value: Any) -> str | None:
    """Testo in chiaro; None se il valore è cifrato e la chiave non è disponibile."""
    if not is_encrypted(value):
        return value
    try:
        return client_encryption().decrypt(value)
    except Exception:
        return None


def stored_view(value: Any) -> dict[str, Any]:
    """Come il valore è salvato in Atlas: per i cifrati, sottotipo, dimensione e inizio dei byte."""
    if not is_encrypted(value):
        return {"encrypted": False, "value": value}
    raw = base64.b64encode(bytes(value)).decode()
    return {"encrypted": True, "subtype": ENCRYPTED_SUBTYPE, "bytes": len(value),
            "base64": raw[:56] + ("…" if len(raw) > 56 else "")}


def self_test() -> str:
    """Verifica completa: chiave master, chiave dati nel key vault, cifratura e decifratura."""
    if not is_configured():
        raise EncryptionUnavailable("Chiave master locale non trovata: la prima domanda delle sessioni resta in chiaro.")
    ensure_data_key()
    sample = "verifica"
    cipher = encrypt_text(sample)
    if not is_encrypted(cipher) or client_encryption().decrypt(cipher) != sample:
        raise RuntimeError("Cifratura e decifratura non coincidono.")
    return "Chiave dati nel key vault, cifratura e decifratura riuscite"


def create_master_key() -> bool:
    """Crea il file della chiave master se manca (permessi solo per l'utente). True se creato."""
    path = config.CSFLE_MASTER_KEY_PATH
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(os.urandom(96))
    return True


def _main(argv: list[str]) -> int:
    if argv[:1] != ["setup"]:
        print("Uso: python -m src.field_encryption setup")
        return 2
    created = create_master_key()
    print(("Creata" if created else "Già presente") + f" la chiave master locale in {config.CSFLE_MASTER_KEY_PATH}")
    ensure_data_key()
    print(f"Chiave dati '{config.CSFLE_KEY_ALT_NAME}' pronta in {config.CSFLE_KEY_VAULT_NAMESPACE}")
    print(self_test())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
