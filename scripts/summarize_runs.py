"""Parse game JSONs under data/runs/ and produce a comparison summary.

For each game we extract:
* impostor / crewmate model
* who used RAG (inferred from filename prefix: ``exp_smoke_*`` = no RAG,
  ``exp_rag_*`` = RAG on the impostor side)
* winner (Crewmates / Impostors / round-limit draw)
* final round number
* kill count
* meetings (vote rounds)

Output: a printed table grouped by matchup, plus a CSV at
``data/exp_run_summary.csv`` for easy sharing.

This is read-only — it never modifies game files or the RAG store.
"""

import csv
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from among_them.config import RUNS_DIR  # noqa: E402

OUT_CSV = "data/exp_run_summary.csv"


def classify_winner(playthrough):
    has_crew = any("Crewmates win!" in line for line in playthrough)
    has_imp = any("Impostors win!" in line for line in playthrough)
    if has_crew and not has_imp:
        return "Crewmates"
    if has_imp and not has_crew:
        return "Impostors"
    return "Draw / Round limit"


def count_kills(playthrough):
    return sum(1 for line in playthrough if re.search(r"eliminated \w+ \(left dead body", line))


def count_body_reports(playthrough):
    return sum(1 for line in playthrough if line.startswith("report:") and "reported a dead body" in line)


def count_banishments(playthrough):
    return sum(1 for line in playthrough if "was banished" in line.lower())


def rag_label(filename):
    if filename.startswith("exp_rag_"):
        return "RAG on impostor"
    if filename.startswith("exp_smoke_"):
        return "no RAG"
    return "(other / unknown)"


def short_model(model: str) -> str:
    s = model.split("/")[-1]
    s = s.replace("-a3b-instruct", "").replace("-it", "")
    return s


def summarize_one(path):
    with open(path) as f:
        g = json.load(f)
    playthrough = g.get("playthrough", [])
    players = g.get("players", [])
    impostor = next((p for p in players if p.get("role") == "Impostor"), None)
    crewmate = next((p for p in players if p.get("role") == "Crewmate"), None)
    return {
        "file": os.path.basename(path),
        "rag": rag_label(os.path.basename(path)),
        "impostor_model": short_model(impostor["llm_model_name"]) if impostor else "?",
        "crewmate_model": short_model(crewmate["llm_model_name"]) if crewmate else "?",
        "winner": classify_winner(playthrough),
        "rounds": g.get("round_number", "?"),
        "kills": count_kills(playthrough),
        "reports": count_body_reports(playthrough),
        "banished": count_banishments(playthrough),
        "impostor_alive": (impostor.get("state", {}).get("life") if impostor else "?"),
    }


def main():
    project_root = os.path.abspath(os.path.join(HERE, ".."))
    os.chdir(project_root)

    files = sorted(
        f for f in os.listdir(RUNS_DIR)
        if f.endswith(".json") and (f.startswith("exp_") or f.startswith("smoke_"))
    )
    if not files:
        print(f"No games found in {RUNS_DIR}")
        return

    rows = []
    for f in files:
        try:
            rows.append(summarize_one(os.path.join(RUNS_DIR, f)))
        except Exception as e:
            print(f"  skipped {f}: {e}")

    # Print table
    hdr = ["file", "rag", "impostor_model", "crewmate_model",
           "winner", "rounds", "kills", "reports", "banished"]
    widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in hdr}
    print()
    print(" | ".join(h.ljust(widths[h]) for h in hdr))
    print("-+-".join("-" * widths[h] for h in hdr))
    for r in rows:
        print(" | ".join(str(r.get(h, "")).ljust(widths[h]) for h in hdr))

    # Write CSV
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr + ["impostor_alive"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV}")

    # Highlight the RAG comparison if both halves exist
    print("\n=== RAG vs no-RAG comparison (same matchup) ===")
    by_matchup = defaultdict(list)
    for r in rows:
        key = (r["impostor_model"], r["crewmate_model"])
        by_matchup[key].append(r)

    for matchup, runs in by_matchup.items():
        if len({r["rag"] for r in runs}) > 1:
            print(f"\nMatchup: {matchup[0]} (imp) vs {matchup[1]} (crew)")
            for r in runs:
                print(
                    f"  [{r['rag']:18s}] winner={r['winner']:18s} "
                    f"rounds={r['rounds']} kills={r['kills']} "
                    f"reports={r['reports']} banished={r['banished']}"
                )


if __name__ == "__main__":
    main()
