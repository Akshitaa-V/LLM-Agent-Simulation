"""Build runs-scoped GUI inputs from the games in ``data/runs/``.

The GUI's Tournaments tab reads two precomputed files:

* ``data/analysis.json``           — per-model technique aggregates.
* ``data/combined_annotations.csv`` — per-utterance examples used in
                                       the "Example Usage of Each
                                       Technique" expanders.

Both default to the legacy ``data/tournament/`` corpus (claude / gpt-4o /
llama / gemini). This script overwrites them with counts and examples
derived only from ``RUNS_DIR``, using the keyword-heuristic taxonomy
from ``scripts/analyze_runs.py`` so the GUI shows the models and
utterances from *our* runs.

Behaviour:
* Walks every ``*.json`` in ``RUNS_DIR``.
* Backs up each existing file once on first run:
    ``data/analysis.json``           -> ``data/analysis_tournaments_backup.json``
    ``data/combined_annotations.csv`` -> ``data/combined_annotations_tournaments_backup.csv``
  so the legacy view can be restored with two ``mv`` commands.
* Token usage is read from the game JSON's per-player ``state.token_usage``
  fields, so the GUI's token chart works too.
"""

import csv
import json
import os
import shutil
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Reuse the technique heuristics + utterance extractor so the GUI view
# stays in sync with the CLI analyser.
from analyze_runs import (  # noqa: E402
    TECHNIQUE_PATTERNS,
    extract_utterances,
    short_model,
)
from among_them.config import RUNS_DIR  # noqa: E402

OUT_PATH = "data/analysis.json"
BACKUP_PATH = "data/analysis_tournaments_backup.json"
OUT_CSV = "data/combined_annotations.csv"
BACKUP_CSV = "data/combined_annotations_tournaments_backup.csv"
OUT_WINS_CSV = "data/persuasion_wins_analysis.csv"
BACKUP_WINS_CSV = "data/persuasion_wins_analysis_tournaments_backup.csv"


def utterance_annotations(playthrough, role_by_name):
    """Yield ``(speaker, role, text, matched_techniques)`` per utterance.

    Filters utterances by known speakers (defends against System rows).
    """
    for name, msg in extract_utterances(playthrough):
        if name not in role_by_name:
            continue
        matched = []
        for technique, patterns in TECHNIQUE_PATTERNS.items():
            if any(p.search(msg) for p in patterns):
                matched.append(technique)
        yield name, role_by_name[name], msg, matched


def back_up_once(src: str, dst: str) -> None:
    """Copy ``src`` to ``dst`` if ``src`` exists and ``dst`` does not."""
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)
        print(f"Backed up existing {src} -> {dst}")


def main() -> None:
    project_root = os.path.abspath(os.path.join(HERE, ".."))
    os.chdir(project_root)

    if not os.path.isdir(RUNS_DIR):
        print(f"Runs dir not found: {RUNS_DIR}")
        return

    files = sorted(f for f in os.listdir(RUNS_DIR) if f.endswith(".json"))
    if not files:
        print(f"No .json files in {RUNS_DIR}")
        return

    back_up_once(OUT_PATH, BACKUP_PATH)
    back_up_once(OUT_CSV, BACKUP_CSV)
    back_up_once(OUT_WINS_CSV, BACKUP_WINS_CSV)

    model_techniques: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    model_player_counts: dict[str, int] = defaultdict(int)
    model_input_tokens: dict[str, dict[str, int]] = defaultdict(dict)
    model_output_tokens: dict[str, dict[str, int]] = defaultdict(dict)
    annotation_rows: list[dict] = []
    wins_rows: list[dict] = []

    n_games = 0
    n_skipped = 0
    for f in files:
        path = os.path.join(RUNS_DIR, f)
        try:
            with open(path) as fp:
                g = json.load(fp)
        except Exception as e:
            print(f"  skipped {f}: {e}")
            n_skipped += 1
            continue

        players = g.get("players", [])
        if not players:
            n_skipped += 1
            continue

        role_by_name = {p["name"]: p["role"] for p in players}
        model_by_name = {p["name"]: p["llm_model_name"] for p in players}

        # Classify outcome for the wins CSV. Skip Unfinished (round_limit)
        # games for the win/loss columns — we can't credit either side.
        playthrough = g.get("playthrough", [])
        crew_won = any("Crewmates win!" in line for line in playthrough)
        imp_won = any("Impostors win!" in line for line in playthrough)
        if crew_won != imp_won:  # exactly one side has a decisive marker
            wins_rows.append({
                "file_name": f,
                "role": "impostor",
                "is_win": imp_won,
            })
            wins_rows.append({
                "file_name": f,
                "role": "crewmate",
                "is_win": crew_won,
            })

        # Per-utterance pass: count techniques per player AND collect
        # annotation rows for the GUI's "Example Usage" expanders.
        per_player_techniques: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for speaker, role, text, matched in utterance_annotations(
            g.get("playthrough", []), role_by_name
        ):
            for t in matched:
                per_player_techniques[speaker][t] += 1
            # We emit one CSV row per utterance even when nothing matched,
            # so the file can be reused for other analyses, but the GUI's
            # display loop filters on non-null annotations.
            annotation_rows.append({
                "text": f"[{speaker}]: {text}",
                "annotation": ";".join(matched).lower() if matched else "",
                "source_file": f,
                "speaker": speaker,
                # GUI shows this as the model badge — short form is more
                # readable than the full uni/... prefix.
                "model": short_model(model_by_name[speaker]),
                "role": role.lower(),
            })

        for p in players:
            model = p["llm_model_name"]
            model_player_counts[model] += 1

            usage = p.get("state", {}).get("token_usage", {}) or {}
            model_input_tokens[model][f] = int(usage.get("input_tokens", 0) or 0)
            model_output_tokens[model][f] = int(usage.get("output_tokens", 0) or 0)

            for technique, count in per_player_techniques.get(p["name"], {}).items():
                model_techniques[model][technique] += count

        n_games += 1

    payload = {
        "model_techniques": {
            m: dict(t) for m, t in model_techniques.items()
        },
        "model_player_counts": dict(model_player_counts),
        "model_input_tokens": {
            m: dict(d) for m, d in model_input_tokens.items()
        },
        "model_output_tokens": {
            m: dict(d) for m, d in model_output_tokens.items()
        },
    }

    with open(OUT_PATH, "w") as fp:
        json.dump(payload, fp, indent=2)

    with open(OUT_CSV, "w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["text", "annotation", "source_file", "speaker", "model", "role"],
        )
        writer.writeheader()
        writer.writerows(annotation_rows)

    with open(OUT_WINS_CSV, "w", newline="") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["file_name", "role", "is_win"]
        )
        writer.writeheader()
        writer.writerows(wins_rows)

    annotated = sum(1 for r in annotation_rows if r["annotation"])
    print()
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {OUT_CSV}  rows={len(annotation_rows)}  with_annotation={annotated}")
    print(f"Wrote {OUT_WINS_CSV}  rows={len(wins_rows)}")
    print(f"  Games indexed:  {n_games}  (skipped {n_skipped})")
    print(f"  Models seen:    {len(model_player_counts)}")
    for m, n in sorted(model_player_counts.items(), key=lambda x: -x[1]):
        uses = sum(model_techniques[m].values())
        print(f"    {m:42s} slots={n:3d}  technique_uses={uses}")


if __name__ == "__main__":
    main()
