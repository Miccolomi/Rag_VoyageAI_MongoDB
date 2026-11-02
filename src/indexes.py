"""Creazione / verifica degli indici Atlas Search a partire dagli schemi JSON.

Supporta quattro modalità di indice vector:
- "auto"    → autoEmbed con Voyage (vettori NON nei documenti)
- "manual"  → vector classico (campo `embedding` calcolato dall'app, dim da .env)
- "context" → vector classico con voyage-context-4 (campo `embedding_context`)
- "multimodal" → vector classico con voyage-multimodal-3.5
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Literal

from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel

from . import config

_SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"

Mode = Literal["auto", "manual", "context", "multimodal"]


def _load(name: str) -> dict[str, Any]:
    with (_SCHEMAS_DIR / name).open() as f:
        return json.load(f)


def _vector_schema(mode: Mode) -> dict[str, Any]:
    if mode == "auto":
        schema = _load("vector_index.json")
        # Allinea il modello a quello in env
        for f in schema["definition"]["fields"]:
            if f.get("type") == "autoEmbed":
                f["model"] = config.VOYAGE_EMBED_MODEL
        return schema
    elif mode == "manual":
        schema = _load("vector_index_manual.json")
        # Allinea numDimensions all'env (deve combaciare con il modello scelto)
        for f in schema["definition"]["fields"]:
            if f.get("type") == "vector":
                f["numDimensions"] = config.VOYAGE_EMBED_DIMENSIONS
        return schema
    elif mode == "context":
        schema = _load("vector_index_context.json")
        # voyage-context-4: 1024 default, allineato all'env
        for f in schema["definition"]["fields"]:
            if f.get("type") == "vector":
                f["path"] = "embedding_context"
                f["numDimensions"] = config.VOYAGE_CONTEXT_DIMENSIONS
        return schema
    elif mode == "multimodal":
        schema = _load("vector_index_multimodal.json")
        for f in schema["definition"]["fields"]:
            if f.get("type") == "vector":
                f["numDimensions"] = config.VOYAGE_MULTIMODAL_DIMENSIONS
        return schema
    raise ValueError(f"mode non valido: {mode}")


def index_specs(mode: Mode) -> list[dict[str, Any]]:
    """Restituisce nomi, tipi e definizioni esatte usate per creare gli indici."""
    vector = _vector_schema(mode)
    vector["name"] = config.vector_index_name(mode)
    text = _load("search_index.json")
    text["name"] = config.SEARCH_INDEX_NAME
    return [
        {"name": vector["name"], "type": "vectorSearch", "definition": vector["definition"]},
        {"name": text["name"], "type": "search", "definition": text["definition"]},
    ]


def _canonical(value: Any) -> Any:
    """Ignora l'ordine dei campi vector, conservando gli ordini degli analyzer."""
    if isinstance(value, dict):
        result = {key: _canonical(item) for key, item in value.items()}
        fields = result.get("fields")
        if isinstance(fields, list) and all(isinstance(field, dict) and "path" in field for field in fields):
            result["fields"] = sorted(fields, key=lambda field: (field["path"], field.get("type", "")))
        return result
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def _differences(actual: Any, expected: Any, path: str = "definition") -> list[str]:
    if actual == expected:
        return []
    if isinstance(actual, dict) and isinstance(expected, dict):
        result = []
        for key in sorted(set(actual) | set(expected)):
            label = f"{path}.{key}"
            if key not in actual:
                result.append(f"{label}: aggiungere {json.dumps(expected[key], ensure_ascii=False)}")
            elif key not in expected:
                result.append(f"{label}: rimuovere {json.dumps(actual[key], ensure_ascii=False)}")
            else:
                result.extend(_differences(actual[key], expected[key], label))
        return result
    if path == "definition.fields" and isinstance(actual, list) and isinstance(expected, list):
        return _differences({field["path"]: field for field in actual},
                            {field["path"]: field for field in expected}, path)
    return [f"{path}: {json.dumps(actual, ensure_ascii=False)} → {json.dumps(expected, ensure_ascii=False)}"]


def index_plan(coll: Collection, *, modes: tuple[Mode, ...] = ("auto", "manual", "context", "multimodal")) -> list[dict[str, Any]]:
    """Confronto in sola lettura; l'indice testuale condiviso compare una volta."""
    specs: dict[str, dict[str, Any]] = {}
    for mode in modes:
        for spec in index_specs(mode):
            if spec["name"] in specs and specs[spec["name"]] != spec:
                raise ValueError(f"Nome indice assegnato a definizioni diverse: {spec['name']}")
            specs[spec["name"]] = spec
    existing = {idx["name"]: idx for idx in coll.list_search_indexes()}
    plan = []
    for spec in specs.values():
        current = existing.get(spec["name"])
        differences = []
        if current is None:
            action = "create"
        elif current.get("type", "search") != spec["type"]:
            action = "blocked"
            differences = [f"Tipo attuale {current.get('type', 'search')}, richiesto {spec['type']}: serve ricreazione esplicita."]
        else:
            definition = current.get("latestDefinition", current.get("definition"))
            if definition is None:
                action = "blocked"
                differences = ["Atlas non restituisce la definizione: impossibile verificare lo schema."]
            else:
                differences = _differences(_canonical(definition), _canonical(spec["definition"]))
                action = "update" if differences else "unchanged"
        plan.append({"spec": spec, "action": action, "differences": differences,
                     "before": current.get("latestDefinition", current.get("definition")) if current else None,
                     "status": current.get("status", "MANCANTE") if current else "MANCANTE",
                     "queryable": bool(current and current.get("queryable"))})
    return plan


def index_command(database: str, collection: str, spec: dict, action: str) -> str:
    target = f"db.getSiblingDB({json.dumps(database)}).getCollection({json.dumps(collection)})"
    definition = json.dumps(spec["definition"], ensure_ascii=False, indent=2)
    if action == "update":
        return f"{target}.updateSearchIndex({json.dumps(spec['name'])}, {definition})"
    return f"{target}.createSearchIndex({json.dumps(spec['name'])}, {json.dumps(spec['type'])}, {definition})"


def ensure_indexes(
    coll: Collection,
    *,
    mode: Mode = "auto",
    recreate: bool = False,
    modes: tuple[Mode, ...] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Crea gli indici mancanti e aggiorna quelli difformi; rende conto per indice.

    Le operazioni Atlas sono asincrone. La ricreazione con drop resta una scelta
    esplicita; il setup normale usa update_search_index e conserva i documenti.
    """
    plan = index_plan(coll, modes=modes or (mode,))
    created, updated, dropped, unchanged, operations = [], [], [], [], []
    for item in plan:
        spec = item["spec"]
        name = spec["name"]
        if on_progress:
            action_label = {"create": "creazione", "update": "aggiornamento della definizione",
                            "unchanged": "definizione già conforme", "blocked": "intervento richiesto"}[item["action"]]
            on_progress(f"{name}: {action_label}.")
        operation = {"name": name, "type": spec["type"], "action": item["action"],
                     "differences": item["differences"], "before": item["before"],
                     "expected": spec["definition"], "status_before": item["status"],
                     "error": None}
        try:
            if recreate and item["status"] != "MANCANTE":
                coll.drop_search_index(name)
                dropped.append(name)
                deadline = time.monotonic() + 60
                while any(idx.get("name") == name for idx in coll.list_search_indexes()):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Rimozione ancora in corso su Atlas. Attendi e riesegui il setup senza Drop & recreate.")
                    time.sleep(0.5)
                operation["action"] = "create"
            if operation["action"] == "blocked":
                raise RuntimeError("; ".join(item["differences"]))
            if operation["action"] == "create":
                coll.create_search_index(SearchIndexModel(
                    definition=spec["definition"], name=name, type=spec["type"],
                ))
                created.append(name)
            elif operation["action"] == "update":
                coll.update_search_index(name, spec["definition"])
                updated.append(name)
            else:
                unchanged.append(name)
            operation["command"] = index_command(coll.database.name, coll.name, spec, operation["action"]) if operation["action"] != "unchanged" else None
        except Exception as exc:
            operation["error"] = f"{type(exc).__name__}: {exc}"
        if on_progress:
            on_progress(f"{name}: " + (operation["error"] or ("verificato, nessuna modifica." if operation["action"] == "unchanged" else "richiesta accettata da Atlas; verificare lo stato di costruzione.")))
        operations.append(operation)
    current = {idx["name"]: idx for idx in coll.list_search_indexes()}
    return {
        "database": coll.database.name, "collection": coll.name,
        "mode": mode if modes is None else "all", "recreate": recreate,
        "created": created, "updated": updated, "dropped": dropped,
        "already_present": unchanged, "operations": operations,
        "atlas_status": {
            item["spec"]["name"]: {
                "status": current.get(item["spec"]["name"], {}).get("status", "NOT_VISIBLE_YET"),
                "queryable": current.get(item["spec"]["name"], {}).get("queryable", False),
                "type": item["spec"]["type"],
            } for item in plan
        },
    }


if __name__ == "__main__":
    print(json.dumps(ensure_indexes(config.get_collection(), modes=("auto", "manual", "context", "multimodal")), default=str, indent=2))
