"""Singleton retrieval layer used by the agents at prompt time.

All public ``get_*_context`` functions return ``""`` on any failure (missing
ChromaDB install, empty store, query exception) so the game never crashes
because of RAG. Callers can therefore inject the return value directly into
a prompt: when retrieval is disabled or broken it simply renders as nothing.
"""

from typing import Optional

try:
    from among_them.rag.store import RAGStore
    from among_them.rag.seeder import seed_strategies
    _CHROMADB_AVAILABLE = True
except Exception:  # ImportError or chromadb runtime errors during import
    RAGStore = None  # type: ignore[assignment]
    seed_strategies = None  # type: ignore[assignment]
    _CHROMADB_AVAILABLE = False


_store: Optional["RAGStore"] = None  # type: ignore[name-defined]


def get_store() -> Optional["RAGStore"]:  # type: ignore[name-defined]
    """Lazily build the singleton store and ensure curated strategies exist."""
    global _store
    if not _CHROMADB_AVAILABLE:
        return None
    if _store is None:
        try:
            _store = RAGStore()
            if not _store.is_seeded():
                seed_strategies(_store)
        except Exception:
            # If ChromaDB itself blows up (corrupt persist dir, missing
            # native deps, ...), disable RAG silently for this process.
            _store = None
    return _store


def get_action_context(role: str, location: str, in_room: str) -> str:
    store = get_store()
    if store is None:
        return ""
    try:
        query = f"{role} in {location} with {in_room}"
        strategies = store.query_strategy(query, role=role.lower(), n=2)
        plays = store.query_action(query, role=role.lower(), n=2)
        parts = []
        if strategies:
            parts.append(
                "--- Proven strategies for your role ---\n"
                + "\n\n".join(strategies)
            )
        if plays:
            parts.append(
                "--- What worked in similar past situations ---\n"
                + "\n\n".join(plays)
            )
        return "\n\n".join(parts)
    except Exception:
        return ""


def get_discussion_context(role: str, recent_messages: str) -> str:
    store = get_store()
    if store is None:
        return ""
    try:
        strategies = store.query_strategy(recent_messages, role=role.lower(), n=2)
        plays = store.query_discussion(recent_messages, role=role.lower(), n=2)
        parts = []
        if strategies:
            parts.append(
                "--- Proven strategies for your role ---\n"
                + "\n\n".join(strategies)
            )
        if plays:
            parts.append(
                "--- Discussion moves that worked in similar situations ---\n"
                + "\n\n".join(plays)
            )
        return "\n\n".join(parts)
    except Exception:
        return ""


def get_voting_context(role: str, discussion_log: str) -> str:
    store = get_store()
    if store is None:
        return ""
    try:
        plays = store.query_discussion(
            discussion_log[-500:], role=role.lower(), n=3
        )
        if not plays:
            return ""
        return (
            "--- Voting patterns from similar past discussions ---\n"
            + "\n\n".join(plays)
        )
    except Exception:
        return ""
