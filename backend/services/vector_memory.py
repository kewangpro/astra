"""
Semantic vector memory backed by ChromaDB.
Stores 'Lessons Learned' with structured metadata for regime-specific retrieval.
"""
from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from backend.config import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "lessons_learned"
_client: chromadb.ClientAPI | None = None
_model: SentenceTransformer | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def _get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_collection() -> chromadb.Collection:
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_lesson(
    lesson_id: str,
    text: str,
    *,
    run_id: str,
    domain: str,
    hyperparameter_name: str | None = None,
    hyperparameter_value: str | None = None,
    environment_config: str | None = None,
    extra: dict | None = None,
) -> None:
    """Store a lesson learned with structured metadata for future retrieval."""
    model = _get_embedding_model()
    embedding = model.encode(text).tolist()

    metadata: dict = {
        "run_id": run_id,
        "domain": domain,
    }
    if hyperparameter_name:
        metadata["hyperparameter_name"] = hyperparameter_name
    if hyperparameter_value is not None:
        metadata["hyperparameter_value"] = str(hyperparameter_value)
    if environment_config:
        metadata["environment_config"] = environment_config
    if extra:
        # ChromaDB metadata values must be str/int/float/bool
        for k, v in extra.items():
            metadata[k] = str(v)

    collection = _get_collection()
    collection.upsert(
        ids=[lesson_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata],
    )
    logger.debug("Stored lesson %s (domain=%s)", lesson_id, domain)


def query_lessons(
    query_text: str,
    *,
    domain: str | None = None,
    environment_config: str | None = None,
    n_results: int = 5,
) -> list[dict]:
    """
    Retrieve the most semantically relevant lessons.
    Optionally filter by domain and/or environment_config for regime-specific retrieval.
    """
    model = _get_embedding_model()
    embedding = model.encode(query_text).tolist()

    where: dict = {}
    if domain and environment_config:
        where = {"$and": [{"domain": domain}, {"environment_config": environment_config}]}
    elif domain:
        where = {"domain": domain}
    elif environment_config:
        where = {"environment_config": environment_config}

    collection = _get_collection()
    kwargs: dict = {"query_embeddings": [embedding], "n_results": n_results, "include": ["documents", "metadatas", "distances"]}
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    lessons = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        lessons.append({"text": doc, "metadata": meta, "distance": dist})
    return lessons


def delete_lesson(lesson_id: str) -> None:
    collection = _get_collection()
    collection.delete(ids=[lesson_id])
    logger.debug("Deleted lesson %s", lesson_id)
