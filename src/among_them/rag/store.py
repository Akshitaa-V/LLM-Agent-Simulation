"""ChromaDB-backed retrieval store for the Among Us RAG system.

Three collections:

* ``strategies``       — curated narrative strategies, keyed by name.
* ``action_plays``     — situation/action/outcome triplets parsed from past games.
* ``discussion_plays`` — discussion-statement examples parsed from past games.

The store is local-only (no API keys, no network) and uses ChromaDB's bundled
``all-MiniLM-L6-v2`` embedding function. All public methods are designed to
``upsert`` so re-seeding is idempotent.
"""

import hashlib
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions


class RAGStore:
    def __init__(self, persist_dir: str = "data/rag_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        ef = embedding_functions.DefaultEmbeddingFunction()
        self.strategies = self.client.get_or_create_collection(
            "strategies", embedding_function=ef
        )
        self.action_plays = self.client.get_or_create_collection(
            "action_plays", embedding_function=ef
        )
        self.discussion_plays = self.client.get_or_create_collection(
            "discussion_plays", embedding_function=ef
        )

    def is_seeded(self) -> bool:
        """True if at least one curated strategy is present."""
        return self.strategies.count() > 0

    def add_strategy(self, text: str, role: str, strategy_name: str) -> None:
        """Upsert a curated strategy. ``strategy_name`` is the stable ID."""
        self.strategies.upsert(
            ids=[strategy_name],
            documents=[text],
            metadatas=[{"role": role, "strategy_name": strategy_name}],
        )

    def add_action_play(
        self,
        situation: str,
        action: str,
        outcome: str,
        role: str,
        won: bool,
        location: str,
        source_file: str = "",
    ) -> None:
        """Upsert an action-phase play. ID is content-hashed for idempotency."""
        uid = hashlib.md5(
            f"{situation}{action}{source_file}".encode("utf-8")
        ).hexdigest()
        doc = (
            f"Situation: {situation}\n"
            f"Action taken: {action}\n"
            f"Result: {outcome}\n"
            f"Outcome: {'WIN' if won else 'LOSS'}"
        )
        self.action_plays.upsert(
            ids=[uid],
            documents=[doc],
            metadatas=[{
                "role": role,
                "won": won,
                "location": location,
                "source_file": source_file,
            }],
        )

    def add_discussion_play(
        self,
        situation: str,
        statement: str,
        techniques: Optional[List[str]],
        role: str,
        won: bool,
        source_file: str = "",
    ) -> None:
        """Upsert a discussion-phase statement. ID is content-hashed."""
        uid = hashlib.md5(
            f"{statement}{source_file}".encode("utf-8")
        ).hexdigest()
        techniques_str = ", ".join(techniques) if techniques else "unknown"
        doc = (
            f"Context: {situation}\n"
            f"Statement: {statement}\n"
            f"Techniques used: {techniques_str}\n"
            f"Outcome: {'WIN' if won else 'LOSS'}"
        )
        self.discussion_plays.upsert(
            ids=[uid],
            documents=[doc],
            metadatas=[{
                "role": role,
                "won": won,
                "source_file": source_file,
            }],
        )

    # ---- query helpers --------------------------------------------------

    def _safe_query(self, collection, query: str, role: str, n: int) -> List[str]:
        """Run a query that never raises — empty results on any error/empty store."""
        try:
            count = collection.count()
            if count == 0:
                return []
            results = collection.query(
                query_texts=[query],
                n_results=min(n, count),
                where={"role": role},
            )
            docs = results.get("documents") or []
            return docs[0] if docs else []
        except Exception:
            return []

    def query_action(self, query: str, role: str, n: int = 3) -> List[str]:
        return self._safe_query(self.action_plays, query, role, n)

    def query_discussion(self, query: str, role: str, n: int = 3) -> List[str]:
        return self._safe_query(self.discussion_plays, query, role, n)

    def query_strategy(self, query: str, role: str, n: int = 2) -> List[str]:
        return self._safe_query(self.strategies, query, role, n)
