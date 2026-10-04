"""RAGAS-style evaluation of the strategy retriever.

Samples real (role, location, in_room) queries from the saved games
we ran, calls the retriever on each, and asks RAGAS to score how
relevant the retrieved strategies are to the query.

RAGAS's core Q&A metrics don't all map to an action-selection RAG
pipeline. We use the two that do:

* ``LLMContextPrecisionWithoutReference`` — for each retrieved chunk,
  the judge decides whether it is useful for answering the question,
  given a synthesised "ideal response". We synthesise the response as
  a short natural-language description of what a competent player in
  that role/location would do next.
* ``ResponseRelevancy`` — measures how well that synthesised response
  addresses the query.

Judge model: ``uni/qwen3-next-80b-a3b-instruct``.
Embeddings for ResponseRelevancy: ``ollama/nomic-embed-text`` if
available, otherwise the metric is skipped.

Writes ``data/exp_rag_eval.csv`` with per-query scores plus prints
the aggregate.
"""

import csv
import json
import os
import random
import re
import sys
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from among_them.config import RUNS_DIR  # noqa: E402
from among_them.llm_factory import make_llm  # noqa: E402
from among_them.rag.retriever import get_store  # noqa: E402

from langchain.schema import HumanMessage, SystemMessage  # noqa: E402

from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import LLMContextPrecisionWithoutReference  # noqa: E402


RELEVANCE_SYSTEM = """You judge whether a retrieved strategy document is CONTEXTUALLY RELEVANT to a player's situation in a text-based Among Us game.

A chunk is relevant if the advice inside would be useful for a player in that role, location, and social context — even if it does not describe the exact action they took. Strategy documents are meant to guide future decisions.

Reply with ONLY one JSON object: {"relevant": true/false, "why": "<one short phrase>"}"""


def _rate_chunk_relevance(judge_llm, query: str, chunk: str) -> Optional[bool]:
    msg = (
        f"Situation: {query}\n\n"
        f"Retrieved chunk:\n{chunk[:1500]}\n"
    )
    reply = judge_llm.invoke([
        SystemMessage(content=RELEVANCE_SYSTEM),
        HumanMessage(content=msg),
    ])
    txt = reply.content
    match = re.search(r"\{[^{}]*\}", txt)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return bool(data.get("relevant"))


JUDGE_MODEL = "uni/qwen3-next-80b-a3b-instruct"
OUT_CSV = "data/exp_rag_eval.csv"
N_SAMPLES = 60
BATCH_PREFIXES = (
    "exp_big_rev_a_", "exp_big_rev_b_",
    "exp_big_a_", "exp_big_b_",
)


def _round_samples_from_game(g: dict) -> list[dict]:
    """Every played round (per player) that has non-trivial context.

    Draws from ``history.rounds`` so we get real mid-game states with
    populated seen_actions and a concrete chosen action.
    """
    out = []
    for p in g.get("players") or []:
        rounds = ((p.get("history") or {}).get("rounds")) or []
        for rd in rounds:
            loc = rd.get("location") or ""
            location = loc.value if hasattr(loc, "value") else loc
            in_room = (rd.get("player_in_room") or "").strip()
            action_result = (rd.get("action_result") or "").strip()
            if not in_room or not action_result or not location:
                continue
            out.append({
                "role": p.get("role") or "Crewmate",
                "location": location,
                "in_room": in_room,
                "name": p.get("name") or "?",
                "action_result": action_result,
            })
    return out


def sample_queries(n: int) -> list[dict]:
    files = sorted(
        f for f in os.listdir(RUNS_DIR)
        if any(f.startswith(p) for p in BATCH_PREFIXES) and f.endswith(".json")
    )
    random.shuffle(files)
    pool: list[dict] = []
    for fn in files:
        if len(pool) >= n * 6:  # oversample so we can pick a balanced subset
            break
        try:
            with open(os.path.join(RUNS_DIR, fn)) as f:
                g = json.load(f)
        except Exception:
            continue
        for s in _round_samples_from_game(g):
            s["source_file"] = fn
            pool.append(s)
    random.shuffle(pool)
    return pool[:n]


def synth_response(role: str, location: str, in_room: str, action_result: str) -> str:
    """Use the actual chosen action from the game as the response.

    ``action_result`` is a string like ``"Task X completed!"`` or
    ``"You [Alice] moved to Medbay"``. That's concrete enough for the
    judge to decide whether a retrieved strategy "supported" it.
    """
    return (
        f"You are a {role} in {location}. Players nearby: {in_room}. "
        f"Chosen action outcome: {action_result}"
    )


def main() -> None:
    os.chdir(os.path.abspath(os.path.join(HERE, "..")))
    random.seed(0)

    store = get_store()
    if store is None:
        print("RAG store unavailable — cannot evaluate.", flush=True)
        return

    samples = sample_queries(N_SAMPLES)
    print(f"Sampled {len(samples)} states from {len(BATCH_PREFIXES)}-prefix batches")

    rows: list[dict] = []
    for s in samples:
        query = f"{s['role']} in {s['location']} with {s['in_room']}"
        # Query the store directly so we get one strategy per chunk
        # instead of the joined+headered blob get_action_context returns.
        strategies = store.query_strategy(query, role=s["role"].lower(), n=2)
        plays = store.query_action(query, role=s["role"].lower(), n=2)
        chunks = [c.strip() for c in (strategies + plays) if c and c.strip()]
        response = synth_response(
            s["role"], s["location"], s["in_room"], s["action_result"],
        )
        rows.append({
            "user_input": query,
            "retrieved_contexts": chunks,
            "response": response,
            "role": s["role"],
            "location": s["location"],
            "source_file": s["source_file"],
        })

    non_empty = [r for r in rows if r["retrieved_contexts"]]
    print(
        f"{len(non_empty)}/{len(rows)} samples have non-empty retrieval "
        f"({len(rows) - len(non_empty)} returned nothing)",
        flush=True,
    )
    if not non_empty:
        print("No retrievable samples — check the RAG store is seeded.")
        return

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=r["user_input"],
            retrieved_contexts=r["retrieved_contexts"],
            response=r["response"],
        )
        for r in non_empty
    ])

    judge = LangchainLLMWrapper(make_llm(JUDGE_MODEL, temperature=0))
    metric = LLMContextPrecisionWithoutReference(llm=judge)

    print(f"Judge: {JUDGE_MODEL}. Running context_precision on "
          f"{len(dataset)} samples...", flush=True)
    result = evaluate(dataset=dataset, metrics=[metric], llm=judge)
    print("\n=== Aggregate ===")
    print(result)

    # Second pass — the semantic-relevance metric that actually fits an
    # action-selection RAG pipeline.
    plain_llm = make_llm(JUDGE_MODEL, temperature=0)
    print("\nScoring semantic relevance per chunk...", flush=True)
    per_sample_relevance = []
    for i, r in enumerate(non_empty):
        verdicts = []
        for chunk in r["retrieved_contexts"]:
            v = _rate_chunk_relevance(plain_llm, r["user_input"], chunk)
            if v is not None:
                verdicts.append(v)
        precision = (sum(verdicts) / len(verdicts)) if verdicts else None
        per_sample_relevance.append(precision)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(non_empty)}", flush=True)

    valid = [x for x in per_sample_relevance if x is not None]
    mean_rel = (sum(valid) / len(valid)) if valid else 0.0
    print(f"Semantic relevance (LLM-judged): mean = {mean_rel:.3f}  "
          f"(n={len(valid)})", flush=True)

    scores = result.scores
    fieldnames = [
        "source_file", "role", "location",
        "user_input", "response", "n_chunks",
        "llm_context_precision_without_reference",
        "semantic_relevance",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r, sc, rel in zip(non_empty, scores, per_sample_relevance):
            w.writerow({
                "source_file": r["source_file"],
                "role": r["role"],
                "location": r["location"],
                "user_input": r["user_input"],
                "response": r["response"],
                "n_chunks": len(r["retrieved_contexts"]),
                "llm_context_precision_without_reference":
                    sc.get("llm_context_precision_without_reference", ""),
                "semantic_relevance": "" if rel is None else round(rel, 3),
            })
    print(f"\nWrote {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
