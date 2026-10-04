"""Per-strategy ablation: test each curated impostor strategy in isolation.

For every strategy in ``CURATED_STRATEGIES`` (impostor-only), runs N games
with that strategy's text injected verbatim into the impostor's prompts —
no RAG retrieval. This lets us measure each strategy's individual effect
rather than the bundled effect from RAG retrieval.

Files are saved as ``data/runs/exp_strat_<strategy_name>_<rep>.json``
so the analyzer can group results by strategy.
"""

import argparse
import os
import sys
import time
import traceback
from random import shuffle

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from among_them.config import RUNS_DIR  # noqa: E402
from among_them.game.game_engine import GameEngine  # noqa: E402
from among_them.game.models.engine import GamePhase  # noqa: E402
from among_them.game.players.ai import AIPlayer  # noqa: E402
from among_them.rag.seeder import CURATED_STRATEGIES  # noqa: E402

IMPOSTOR_MODEL = "uni/qwen36-35b"
CREWMATE_MODEL = "uni/qwen3-next-80b-a3b-instruct"
N_PLAYERS = 5
N_IMPOSTORS = 1
ROUND_CAP = 20


def impostor_strategies():
    """Yield (name, text) tuples for every impostor-side curated strategy."""
    for s in CURATED_STRATEGIES:
        if s.get("role", "").lower() == "impostor":
            yield s["name"], s["text"]


def build_players(injected_strategy: str):
    names = ["Alice", "Bob", "Charlie", "Dave", "Eren"][:N_PLAYERS]
    shuffle(names)
    players = []
    for i in range(N_PLAYERS):
        if i < N_IMPOSTORS:
            players.append(
                AIPlayer(
                    name=names[i],
                    llm_model_name=IMPOSTOR_MODEL,
                    role="Impostor",
                    use_rag=False,
                    injected_strategy=injected_strategy,
                )
            )
        else:
            players.append(
                AIPlayer(
                    name=names[i],
                    llm_model_name=CREWMATE_MODEL,
                    role="Crewmate",
                    use_rag=False,
                    injected_strategy="",
                )
            )
    return players


def add_flag(full_path: str, flag: str) -> None:
    new_path = full_path.split(".")[0] + "_" + flag + ".json"
    if os.path.exists(full_path):
        os.rename(full_path, new_path)


def run_one(out_file: str, strategy_text: str) -> dict:
    full_path = os.path.join(RUNS_DIR, out_file)
    players = build_players(strategy_text)
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
        "error": str(err) if err else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a per-strategy ablation batch."
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Games per strategy (default 3).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated strategy names to run; default = all impostor strategies.",
    )
    parser.add_argument(
        "--start-rep",
        type=int,
        default=1,
        help="First rep number (default 1). Use higher to add more reps to an existing batch.",
    )
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(HERE, ".."))
    os.chdir(project_root)
    os.makedirs(RUNS_DIR, exist_ok=True)

    strategies = list(impostor_strategies())
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        strategies = [(n, t) for n, t in strategies if n in wanted]
        missing = wanted - {n for n, _ in strategies}
        if missing:
            print(f"WARNING: unknown strategies: {sorted(missing)}", flush=True)

    if not strategies:
        print("No strategies selected. Exiting.")
        return

    n_total = len(strategies) * args.reps
    est_min = n_total * 6.5
    print(
        f"Strategies: {len(strategies)} | Reps each: {args.reps} | "
        f"Total games: {n_total}"
    )
    print(f"Estimated runtime: {est_min:.0f} min ({est_min / 60:.1f} h)")
    print(f"Impostor: {IMPOSTOR_MODEL} (single strategy injected)")
    print(f"Crewmate: {CREWMATE_MODEL}\n", flush=True)

    batch_t0 = time.time()
    completed = 0
    for s_idx, (name, text) in enumerate(strategies, start=1):
        print(f"\n{'#' * 70}")
        print(f"# STRATEGY {s_idx}/{len(strategies)}: {name}")
        print(f"{'#' * 70}", flush=True)
        for rep in range(args.start_rep, args.start_rep + args.reps):
            out_file = f"exp_strat_{name}_{rep}.json"
            completed += 1
            print(
                f"\n[{name} rep {rep} | overall {completed}/{n_total}] -> {out_file}",
                flush=True,
            )
            r = run_one(out_file, text)
            msg = (
                f"[{name} rep {rep}] done rounds={r['rounds']} "
                f"in {r['elapsed_min']:.1f} min  "
                f"(batch elapsed: {(time.time() - batch_t0) / 60.0:.1f} min)"
            )
            if r["error"]:
                msg += f"  ERROR: {r['error']}"
            print(msg, flush=True)

    total = (time.time() - batch_t0) / 60.0
    print(f"\nAblation finished in {total:.1f} min total")
    print(f"Avg per game: {total / n_total:.1f} min")


if __name__ == "__main__":
    main()
