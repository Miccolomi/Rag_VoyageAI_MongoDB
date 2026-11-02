"""Prepara le collection applicative e gli indici locali GridFS senza rimuovere dati.

Eseguire dalla radice del progetto: python -m src.init_db
I nomi GridFS sono derivati da MONGODB_GRIDFS_BUCKET, come nel driver.
Gli indici Atlas Search sono gestiti separatamente da src.indexes.
"""
from pymongo import ASCENDING
from pymongo.errors import CollectionInvalid
from . import config


def ensure_document_lookup_index(collection):
    """Index document identity and ingestion mode, preserving existing indexes."""
    expected = [("document_id", ASCENDING), ("embedding_mode", ASCENDING)]
    for name, spec in collection.index_information().items():
        if (list(spec.get("key", []))[:2] == expected
                and not spec.get("sparse") and not spec.get("partialFilterExpression")):
            return {"name": name, "created": False}
    name = collection.create_index(expected, name="document_id_1_embedding_mode_1")
    return {"name": name, "created": True}


def repair_document_page_references(collection):
    """Rimuove solo numeri sintetici da estratti a livello documento preesistenti.

    I chunk office_structure derivano dal documento intero; il creator
    MarkItDown identifica il fallback privo di paginazione. Testo e vettori
    restano invariati, così non servono chiamate ai modelli o nuova ingestion.
    """
    structure = collection.update_many(
        {"content_type": "office_structure", "$or": [
            {field: {"$exists": True}} for field in ("page_number", "page_start", "page_end")
        ]},
        {"$unset": {field: "" for field in ("page_number", "page_start", "page_end")}},
    )
    fallback = collection.update_many(
        {"creator": "MarkItDown", "$or": [
            {field: {"$exists": True}} for field in ("page_number", "page_start", "page_end", "total_pages")
        ]},
        {"$unset": {field: "" for field in ("page_number", "page_start", "page_end", "total_pages")}},
    )
    return {"office_structure_corrected": structure.modified_count,
            "markitdown_fallback_corrected": fallback.modified_count}


def initialize_database():
    db = config.get_db()
    bucket = config.MONGODB_GRIDFS_BUCKET
    names = [config.MONGODB_COLLECTION, f"{bucket}.files", f"{bucket}.chunks"]
    created = []
    existing = []
    for name in names:
        try:
            db.create_collection(name)
            created.append(name)
        except CollectionInvalid:
            existing.append(name)
    document_lookup_index = ensure_document_lookup_index(db[config.MONGODB_COLLECTION])
    gridfs_files_indexes = db[f"{bucket}.files"].index_information()
    db[f"{bucket}.files"].create_index(
        [("filename", ASCENDING), ("uploadDate", ASCENDING)],
        name="filename_1_uploadDate_1")
    db[f"{bucket}.chunks"].create_index(
        [("files_id", ASCENDING), ("n", ASCENDING)],
        unique=True, name="files_id_1_n_1")
    collection_stats = {
        name: db[name].estimated_document_count() for name in names
    }
    return {
        "database": db.name,
        "created": created,
        "already_present": existing,
        "collections": names,
        "document_counts_estimated": collection_stats,
        "document_lookup_index": document_lookup_index,
        "gridfs_indexes": {
            "files": list(db[f"{bucket}.files"].index_information()),
            "chunks": list(db[f"{bucket}.chunks"].index_information()),
            "files_indexes_before_setup": sorted(gridfs_files_indexes),
        },
    }


if __name__ == "__main__":
    print(initialize_database())
