"""Model backends for non-generative decisions, text generation, and embeddings."""

from trace_tree_eval.backends.base import DecisionBackend, EmbeddingBackend, TextModelBackend
from trace_tree_eval.backends.cache import DiskCache
from trace_tree_eval.backends.embeddings import (
    OpenAICompatEmbeddingBackend,
    SentenceTransformerEmbeddingBackend,
    create_embedding_backend,
)
from trace_tree_eval.backends.jev import JevBackend
from trace_tree_eval.backends.mock import (
    HashingEmbeddingBackend,
    MockDecisionBackend,
    MockTextBackend,
)
from trace_tree_eval.backends.openai_compat import OpenAICompatTextBackend

__all__ = [
    "DecisionBackend",
    "TextModelBackend",
    "EmbeddingBackend",
    "DiskCache",
    "JevBackend",
    "OpenAICompatTextBackend",
    "SentenceTransformerEmbeddingBackend",
    "OpenAICompatEmbeddingBackend",
    "create_embedding_backend",
    "HashingEmbeddingBackend",
    "MockDecisionBackend",
    "MockTextBackend",
]
