"""
Recipe Library — Step 5.2.

ChromaDB-backed semantic index over crystallized recipes.
Provides warm-start retrieval: given a new goal, find the most relevant
past recipe and surface its hyperparameters to the LeadAgent as a hint.
"""
from __future__ import annotations

import json
from typing import Optional, TYPE_CHECKING

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from backend.config import settings
from backend.logging_config import get_logger

if TYPE_CHECKING:
    from backend.models.recipe import RecipeRecord

logger = get_logger(__name__)

COLLECTION_NAME = "recipe_library"
_client: Optional[chromadb.ClientAPI] = None
_model: Optional[SentenceTransformer] = None


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
        logger.info("RecipeLibrary: loading embedding model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_collection() -> chromadb.Collection:
    return _get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _recipe_text(recipe: "RecipeRecord") -> str:
    """Build a rich text representation for embedding."""
    parts = [
        f"Domain: {recipe.domain}",
        f"Task: {recipe.task_type}",
        f"Description: {recipe.description or ''}",
        f"Algorithm: {recipe.full_content.get('hyperparameters', {}).get('algorithm', '')}",
        f"Hyperparameters: {json.dumps(recipe.hyperparameters)}",
    ]
    if recipe.target_metric:
        parts.append(f"Target metric: {json.dumps(recipe.target_metric)}")
    return "\n".join(parts)


def index_recipe(recipe: "RecipeRecord") -> None:
    """Embed and store a recipe in the semantic index."""
    text = _recipe_text(recipe)
    embedding = _get_embedding_model().encode(text).tolist()
    metadata = {
        "name": recipe.name,
        "domain": recipe.domain,
        "task_type": recipe.task_type,
        "generation": recipe.generation,
        "is_golden": str(recipe.is_golden),
        "score": str(recipe.score or ""),
    }
    _get_collection().upsert(
        ids=[recipe.id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata],
    )
    logger.debug("RecipeLibrary: indexed recipe=%s (domain=%s)", recipe.name, recipe.domain)


def search_recipes(
    query: str,
    *,
    domain: Optional[str] = None,
    n_results: int = 5,
) -> list[dict]:
    """
    Semantic search over the recipe index.
    Returns list of {id, name, domain, score, distance} dicts.
    """
    embedding = _get_embedding_model().encode(query).tolist()
    where: dict = {}
    if domain:
        where = {"domain": domain}

    collection = _get_collection()
    try:
        count = collection.count()
    except Exception:
        count = 0
    if count == 0:
        return []

    kwargs: dict = {
        "query_embeddings": [embedding],
        "n_results": min(n_results, count),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    try:
        results = collection.query(**kwargs)
    except Exception as exc:
        logger.warning("RecipeLibrary: search failed: %s", exc)
        return []

    hits = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        hits.append({
            "id": meta.get("name"),
            "name": meta.get("name"),
            "domain": meta.get("domain"),
            "is_golden": meta.get("is_golden") == "True",
            "distance": dist,
        })
    return hits


def get_warm_start_hint(goal: str, domain: str) -> Optional[dict]:
    """
    Return the hyperparameters of the best matching past recipe as a
    warm-start hint for the LeadAgent, or None if the index is empty.
    """
    hits = search_recipes(goal, domain=domain, n_results=1)
    if not hits:
        return None
    return {"best_matching_recipe": hits[0]["name"], "distance": hits[0]["distance"]}


def remove_recipe(recipe_id: str) -> None:
    _get_collection().delete(ids=[recipe_id])
