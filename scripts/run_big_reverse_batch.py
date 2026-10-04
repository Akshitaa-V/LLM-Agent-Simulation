"""Reverse-matchup version of the big batch.

Same setup as ``run_big_strategy_batch.py`` (3 impostor strategies, 3
Reddit crew strategies, 10 players, 40 round cap), but the models are
swapped: qwen36-35b is now the impostor, gemma4-31b-it is the crew.

Files:
    data/runs/exp_big_rev_a_<imp_strategy>_<rep>.json   (default crew)
    data/runs/exp_big_rev_b_<imp_strategy>_<rep>.json   (reddit crew)
"""

import argparse
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from random import Random

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from among_them.config import RUNS_DIR  # noqa: E402
from among_them.game.game_engine import GameEngine  # noqa: E402
from among_them.game.models.engine import GamePhase  # noqa: E402
from among_them.game.players.ai import AIPlayer  # noqa: E402
from among_them.rag.seeder import CREWMATE_STRATEGIES, CURATED_STRATEGIES  # noqa: E402

IMPOSTOR_MODEL = "uni/qwen36-35b"
CREWMATE_MODEL = "uni/gemma4-31b-it"
N_PLAYERS = 10
N_IMPOSTORS = 2
ROUND_CAP = 40
REPS_PER_STRATEGY = 20
PARALLELISM = 4

TOP_IMPOSTOR_STRATEGIES = (
    "post_kill_alibi",
    "realistic_fake_tasks",
    "silent_hunter",
)
REDDIT_CREWMATE_STRATEGIES = (
    "buddy_system",
    "skeptical_voter",
    "behavioral_profiler",
)

PLAYER_NAME_POOL = [
    "Alice", "Bob", "Charlie", "Dave", "Eren",
    "Modi", "Macroon", "Doraemon", "Ian", "Judy",
    "Kevin", "Liam", "Mona", "Nina", "Oscar",
]


def _lookup(strategies, name):
    for s in strategies:
        if s["name"] == name:
            return s["text"]
    raise KeyError(f"strategy {name!r} not in seed list")


def build_players(rng, imp_strategy_text, crew_strategy_texts):
    """Build a 10-player roster.

    ``crew_strategy_texts`` is a list of strategy strings; each crewmate
    gets one in round-robin order. Pass ``[""]`` for an unstrategized crew.
    """
    names = list(PLAYER_NAME_POOL)
    rng.shuffle(names)
    names = names[:N_PLAYERS]

    players = []
    for i in range(N_PLAYERS):
        if i < N_IMPOSTORS:
            players.append(
                AIPlayer(
                    name=names[i],
                    llm_model_name=IMPOSTOR_MODEL,
                    role="Impostor",
                    use_rag=False,
                    injected_strategy=imp_strategy_text,
                )
            )
        else:
            crew_idx = (i - N_IMPOSTORS) % len(crew_strategy_texts)
            players.append(
                AIPlayer(
                    name=names[i],
                    llm_model_name=CREWMATE_MODEL,
                    role="Crewmate",
                    use_rag=False,
                    injected_strategy=crew_strategy_texts[crew_idx],
                )
            )
    return players


def add_flag(full_path: str, flag: str) -> None:
    new_path = full_path.split(".")[0] + "_" + flag + ".json"
    if os.path.exists(full_path):
        os.rename(full_path, new_path)


def run_one(out_file, imp_text, crew_texts, seed) -> dict:
    full_path = os.path.join(RUNS_DIR, out_file)
    rng = Random(seed)
    players = build_players(rng, imp_text, crew_texts)
    engine = GameEngine(file_path=full_path)
    engine.state.DEBUG = False
    engine.load_players(players, impostor_count=N_IMPOSTORS)
    engine.state.game_stage = GamePhase.ACTION_PHASE

    t0 = time.time()
    finished = False
    err = None
    while not finished and engine.state.round_number < ROUND_CAP:
        try:
            finished = engine.perform_step()
        except Exception as e:
            if "LLM did" in str(e):
                continue
            err = e
            try:
                add_flag(full_path, "exception")
            except Exception:
                pass
            with open("data/error_log.txt", "a") as f:
                f.write(f"{out_file} errored with:\n{e}\n")
            traceback.print_exc()
            break

    if not err and engine.state.round_number >= ROUND_CAP:
        add_flag(full_path, "round_limit")

    return {
        "file": out_file,
        "elapsed_min": (time.time() - t0) / 60.0,
        "rounds": engine.state.round_number,
        "kills": sum(
            1 for p in engine.state.players if p.state.life.value != "Alive"
        ),
        "error": str(err) if err else None,
    }


def _is_finished(out_file: str) -> bool:
    """Skip-existing logic for resume after a crash.

    A game is treated as finished if its file already exists with the
    ``_round_limit`` suffix, OR if the file exists and its playthrough
    contains a decisive win marker. Anything else (partial mid-game
    state from a killed process) is considered unfinished and gets
    re-run.
    """
    full = os.path.join(RUNS_DIR, out_file)
    stem = full[: -len(".json")]
    if os.path.exists(stem + "_round_limit.json"):
        return True
    if not os.path.exists(full):
        return False
    try:
        with open(full) as f:
            payload = f.read()
        return "Impostors win!" in payload or "Crewmates win!" in payload
    except OSError:
        return False


def build_jobs(batch_label, crew_texts, reps):
    jobs = []
    skipped = 0
    for imp_name in TOP_IMPOSTOR_STRATEGIES:
        imp_text = _lookup(CURATED_STRATEGIES, imp_name)
        for rep in range(1, reps + 1):
            out_file = f"exp_big_rev_{batch_label}_{imp_name}_{rep}.json"
            if _is_finished(out_file):
                skipped += 1
                continue
            seed = hash((batch_label, imp_name, rep)) & 0xFFFFFFFF
            jobs.append((out_file, imp_text, crew_texts, seed))
    if skipped:
        print(f"  (skipping {skipped} already-finished games in batch {batch_label!r})", flush=True)
    return jobs


def run_batch(label, jobs):
    print(
        f"\n{'=' * 70}\nBATCH {label.upper()} — {len(jobs)} games "
        f"({PARALLELISM}-way parallel)\n{'=' * 70}",
        flush=True,
    )
    t0 = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=PARALLELISM) as ex:
        futures = {ex.submit(run_one, *j): j[0] for j in jobs}
        for fut in as_completed(futures):
            out_file = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"file": out_file, "error": str(e), "rounds": 0,
                     "elapsed_min": 0, "kills": 0}
            completed += 1
            msg = (
                f"[{label} {completed}/{len(jobs)}] {out_file} -> "
                f"rounds={r['rounds']} kills={r['kills']} "
                f"in {r['elapsed_min']:.1f} min "
                f"(batch elapsed {(time.time() - t0) / 60.0:.1f} min)"
            )
            if r.get("error"):
                msg += f" ERROR: {r['error'][:120]}"
            print(msg, flush=True)
    print(
        f"\nBATCH {label.upper()} finished in {(time.time() - t0) / 60.0:.1f} min",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reps", type=int, default=REPS_PER_STRATEGY,
        help=f"Reps per impostor strategy (default {REPS_PER_STRATEGY}); "
             "60-game batch = 20 reps x 3 strategies.",
    )
    parser.add_argument(
        "--only-a", action="store_true",
        help="Run only Batch A (no crewmate strategies).",
    )
    parser.add_argument(
        "--only-b", action="store_true",
        help="Run only Batch B (Reddit crewmate strategies).",
    )
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(HERE, ".."))
    os.chdir(project_root)
    os.makedirs(RUNS_DIR, exist_ok=True)

    print(f"Impostor: {IMPOSTOR_MODEL}")
    print(f"Crewmate: {CREWMATE_MODEL}")
    print(
        f"{N_PLAYERS} players ({N_IMPOSTORS} impostor) | "
        f"round cap {ROUND_CAP} | parallelism {PARALLELISM}"
    )
    print(f"Impostor strategies: {list(TOP_IMPOSTOR_STRATEGIES)}")
    print(f"Reddit crew strategies: {list(REDDIT_CREWMATE_STRATEGIES)}")
    print(f"Reps per impostor strategy: {args.reps}\n", flush=True)

    batch_t0 = time.time()

    if not args.only_b:
        jobs_a = build_jobs("a", [""], args.reps)
        run_batch("a", jobs_a)

    if not args.only_a:
        crew_texts = [
            _lookup(CREWMATE_STRATEGIES, n) for n in REDDIT_CREWMATE_STRATEGIES
        ]
        jobs_b = build_jobs("b", crew_texts, args.reps)
        run_batch("b", jobs_b)

    total = (time.time() - batch_t0) / 60.0
    print(f"\nAll batches finished in {total:.1f} min", flush=True)

    print("Regenerating analyzer CSVs...", flush=True)
    try:
        subprocess.run(
            [sys.executable, os.path.join(HERE, "analyze_runs.py")],
            check=True,
        )
    except Exception as e:
        print(f"Analyzer failed: {e}", flush=True)


if __name__ == "__main__":
    main()
