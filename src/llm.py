"""LLM factory.

Single place that constructs OpenAI chat and embedding clients. Every node
imports from here so that swapping models, temperatures, or providers is a
one-file change.

Also wires LangSmith tracing on import if ``LANGSMITH_TRACING`` is enabled
in ``.env``. LangChain reads tracing configuration from process env vars,
so we just mirror ``settings`` into ``os.environ`` with both the new
(``LANGSMITH_*``) and legacy (``LANGCHAIN_*``) names so any version of the
SDK picks it up.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import settings


# ----------------------------------------------------------------------------
#  LangSmith tracing (optional)
# ----------------------------------------------------------------------------
def _configure_langsmith() -> None:
    """Mirror LangSmith settings into process env vars."""
    if not settings.langsmith_tracing:
        # Make sure stale values from a parent shell don't accidentally enable
        # tracing when the user set LANGSMITH_TRACING=false in .env.
        os.environ.pop("LANGSMITH_TRACING", None)
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project


_configure_langsmith()


# ----------------------------------------------------------------------------
#  Chat model
# ----------------------------------------------------------------------------
@lru_cache(maxsize=8)
def get_llm(temperature: float = 0.0, max_tokens: int = 2048) -> BaseChatModel:
    """Return a cached ChatOpenAI client for the configured model.

    ``temperature=0`` by default because SQL generation, routing, and
    self-correction all benefit from deterministic outputs. Nodes that
    need creativity (e.g. the NL summary in the formatter) can request a
    higher temperature and will get a separately cached instance.
    """
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=60,
    )


# ----------------------------------------------------------------------------
#  Embeddings (for the Golden SQL few-shot store)
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return a cached OpenAI embeddings client."""
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )


# ----------------------------------------------------------------------------
#  Manual sanity check: `python -m src.llm`
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Chat model:       {settings.openai_model}")
    print(f"Embedding model:  {settings.openai_embedding_model}")
    print(f"LangSmith:        {'enabled' if settings.langsmith_tracing else 'disabled'}")

    print("\n--- chat ping ---")
    reply = get_llm().invoke("Reply with exactly the word: pong")
    print(f"LLM reply: {reply.content!r}")

    print("\n--- embedding ping ---")
    vec = get_embeddings().embed_query("hello world")
    print(f"Embedding dims: {len(vec)} (expected 1536)")
    print(f"First 5 values: {vec[:5]}")
