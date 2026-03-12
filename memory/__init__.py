"""
memory package
---------------
Memory storage modules for AgeMem.
"""

from memory.vector_index import (
    ensure_table_exists,
    insert_embedding,
    update_embedding,
    delete_embedding,
    query_similar,
    get_embedding_count,
    entry_exists,
)

from memory.retrieval import (
    retrieve_relevant_ltm,
    retrieve_by_tags,
    retrieve_recent,
)

from memory.embedding import (
    EmbeddingModule,
    embed_text,
    embed_batch,
    cosine_similarity,
)

# QUERY_EXPANSION: Export QueryExpander for external use
from tools.query_expansion import QueryExpander

__all__ = [
    # Vector index
    "ensure_table_exists",
    "insert_embedding",
    "update_embedding",
    "delete_embedding",
    "query_similar",
    "get_embedding_count",
    "entry_exists",
    # Retrieval
    "retrieve_relevant_ltm",
    "retrieve_by_tags",
    "retrieve_recent",
    # Embedding
    "EmbeddingModule",
    "embed_text",
    "embed_batch",
    "cosine_similarity",
    # Query expansion
    "QueryExpander",
]
