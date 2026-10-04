# Among Them

Fork of a research repo that runs Among Us-style games between LLM agents. Text-based, no visuals — agents pick actions, discuss, and vote each round while their prompts and outputs get saved for analysis. I picked this up as a lab project and my work has been mostly around benchmarking (which prompt strategies actually work, which models actually execute them) plus a bunch of tooling and bug fixes along the way.

![overview.png](images/overview.png)

## What's in here

- Streamlit GUI to run games interactively and browse saved runs.
- Multi-backend LLM support: OpenRouter, University of Passau InnKube hub (the `uni/*` prefix), and local Ollama.
- Game engine that saves state to JSON after every step so anything can be replayed.
- A batch runner for each experiment I've done (`scripts/run_*.py`) plus an analyzer that walks every saved game and writes CSVs the GUI reads.
- Curated RAG strategy library (15 impostor strategies + 6 crewmate strategies) that can either be retrieved by the vector store or injected verbatim into the prompt for ablation.
- LLM-as-a-judge scoring every game on strategy adherence, deception, and crew defense.
- A RAGAS-style evaluation for the RAG pipeline.
- A ruff hook so I stop shipping unformatted code.

![diagram.png](images/diagram.png)

## Directory layout

- `src/among_them/game/` — engine, players, agents.
- `src/among_them/rag/` — Chroma store + seed strategies.
- `src/among_them/gui_handler.py` — the Streamlit views.
- `src/among_them/llm_factory.py` — picks the backend based on the model prefix and now also sets a per-call timeout so a system sleep can't lock the client forever.
- `scripts/` — batch runners, judge, RAG eval, analyzer.
- `data/runs/` — one JSON per game.

## Install

Python 3.11+.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install ruff ragas
```

Poetry also works (`poetry install`) if that's already on your path. Ruff and RAGAS aren't in the pyproject dependencies yet; install them separately.

## Configure

Copy `.env.example` to `.env` and set whichever backend you use:

```
OPENROUTER_API_KEY=...
UNI_API_KEY=...
UNI_BASE_URL=https://llms.innkube.fim.uni-passau.de/v1
OLLAMA_BASE_URL=http://localhost:11434/v1
```

Model names use a prefix to pick the backend: `openai/*` and `anthropic/*` go to OpenRouter, `uni/*` to InnKube, `ollama/*` to local Ollama.

## Run

The GUI:

```bash
.venv/bin/streamlit run src/among_them/main.py
```

Batches (each writes to `data/runs/` and reruns the analyzer at the end):

```bash
# 10 games, one per curated impostor strategy, 5p / 1i / 20-round cap.
.venv/bin/python scripts/run_gemma_strategy_batch.py

# 120 games — top-3 impostor strategies × 20 reps × 2 crew conditions.
# 10 players (2 impostors), 40-round cap, 4-way parallel.
.venv/bin/python scripts/run_big_strategy_batch.py

# Same layout but models swapped.
.venv/bin/python scripts/run_big_reverse_batch.py
```

If you're going to leave a batch running unattended, wrap it in `caffeinate -i` and **keep the laptop lid open** — macOS suspends the network stack when the lid is closed regardless of caffeinate. Lost a night's worth of games learning this one.

Refresh CSVs manually:

```bash
.venv/bin/python scripts/analyze_runs.py
```

Score every game with the LLM judge:

```bash
.venv/bin/python scripts/judge_games.py
```

Verify the RAG pipeline:

```bash
.venv/bin/python scripts/rag_eval.py
```

## Tests

```bash
poetry run pytest
```

## Results

Numbers below are on decisive games only — round-cap games are counted separately.

### Small ablation

5 players, 1 impostor, 20-round cap. Ran one game per curated impostor strategy to shortlist which ones to test at scale.

| Strategy | Winner | Rounds | Kills |
|---|---|---:|---:|
| post_kill_alibi | Impostor | 7 | 1 |
| realistic_fake_tasks | Impostor | 8 | 2 |
| consistent_persona | Crew | 8 | 1 |
| patient_first_kill | Crew | 8 | 1 |
| second_kill_caution | Crew | 6 | 1 |
| silent_hunter | Round cap | 20 | 2 |
| afk_ghost, task_mimic, isolated_kill_rooms, redirector | Round cap | 20 | 0 |

Only two strategies produced impostor wins outright; `silent_hunter` was the only cap-out that killed anyone. Picked top-3 by wins+kills: `post_kill_alibi`, `realistic_fake_tasks`, `silent_hunter`.

### Big batch (240 games total)

10 players, 2 impostors, 40-round cap. 20 reps per strategy × 4 conditions (2 model matchups × 2 crew conditions).

Forward — gemma4-31b impostor vs qwen36-35b crew:

| Strategy | Default crew | Reddit crew | Δ |
|---|---:|---:|---:|
| post_kill_alibi | 66.7% | 45.0% | −22pp |
| realistic_fake_tasks | 65.0% | 20.0% | −45pp |
| silent_hunter | 26.7% | 40.0% | +13pp |

Reverse — qwen36-35b impostor vs gemma4-31b crew:

| Strategy | Default crew | Reddit crew | Δ |
|---|---:|---:|---:|
| post_kill_alibi | 10.0% | 5.0% | −5pp |
| realistic_fake_tasks | 5.3% | 0.0% | −5pp |
| silent_hunter | 0.0% | 0.0% | 0pp |

### What I read into this

- **The model matters more than the prompt.** Same strategies, same setup, swap the impostor model and win rate falls off a cliff (65% → 5% for `realistic_fake_tasks`). Prompt engineering can only get you as far as the model is willing to execute it.
- **Alibi-engineering strategies win, action-gating strategies stall.** `post_kill_alibi` and `realistic_fake_tasks` tell the model what to *say later*; both won. `silent_hunter`, `afk_ghost`, `isolated_kill_rooms`, `redirector` all just tell the model *when* to kill and mostly cap out with no kills committed.
- **Reddit crew strategies hard-counter fake tasks but only soften alibis.** `realistic_fake_tasks` drops 65% → 20% against `task_verifier`-style pressure. `post_kill_alibi` only softens 67% → 45% — "I was with X and Y" survives.
- **`silent_hunter` gets *better* against strategic crews.** Went 27% → 40%. The `behavioral_profiler` "call out anyone who suddenly went quiet" trap doesn't fire on an impostor that was already silent.
- **12 games in the reverse batch errored out** on a filesystem permission for the shared `data/error_log.txt`. `rev_b:silent_hunter` has only 8 games because of it — the 0% is directionally right but not statistically firm.

### Judge scores (all 228 games)

Judge model: `uni/qwen3-next-80b-a3b-instruct`. Scores 1-5 on each dimension, means shown.

| Setup | Strategy | Adherence | Deception | Crew defense |
|---|---|---:|---:|---:|
| gemma4 imp / default crew | post_kill_alibi | 5.00 | 5.00 | 3.65 |
| gemma4 imp / default crew | realistic_fake_tasks | 5.00 | 5.00 | 4.40 |
| gemma4 imp / default crew | silent_hunter | 5.00 | 5.00 | 3.80 |
| gemma4 imp / reddit crew | post_kill_alibi | 5.00 | 4.95 | 3.75 |
| gemma4 imp / reddit crew | realistic_fake_tasks | 4.85 | 4.70 | 4.60 |
| gemma4 imp / reddit crew | silent_hunter | 4.85 | 4.90 | 3.90 |
| qwen36 imp / default crew | post_kill_alibi | 3.70 | 3.80 | 4.65 |
| qwen36 imp / default crew | realistic_fake_tasks | 4.00 | 3.90 | 4.70 |
| qwen36 imp / default crew | silent_hunter | 5.00 | 4.45 | 5.00 |
| qwen36 imp / reddit crew | post_kill_alibi | 4.05 | 4.00 | 4.45 |
| qwen36 imp / reddit crew | realistic_fake_tasks | 4.80 | 4.50 | 4.70 |
| qwen36 imp / reddit crew | silent_hunter | 5.00 | 4.88 | 4.62 |

The scores explain the win-rate collapse in the reverse batch. gemma4-as-impostor gets 4.85-5.0 on adherence and deception. qwen36-as-impostor drops to 3.7-4.0 on both. Same prompt text, worse execution. `silent_hunter` is the outlier — qwen36 scores 5.0 on adherence because "stay quiet and be vague" is trivially easy to follow — but deception is still lower because there's no positive alibi being built.

### RAG pipeline check

`scripts/rag_eval.py` samples 60 real (role, location, in_room) queries from the saved games and scores each retrieval two ways. Judge is the same 80B model.

| Metric | Score |
|---|---:|
| RAGAS `LLMContextPrecisionWithoutReference` | 0.06 |
| Semantic relevance (LLM-judged) | 0.76 |

RAGAS scores near-zero because its Q&A framing asks "does this chunk directly support the response?" and strategy documents don't work that way — they inform future decisions. The semantic-relevance metric ("is this retrieved strategy contextually relevant to a $role in $location with these players nearby?") is closer to what the pipeline is actually trying to do. 76% is OK-not-great; the failure cases cluster on crewmates in peripheral rooms (Reactor, O2) where the seed set is thin. Expanding `CREWMATE_STRATEGIES` in [`seeder.py`](src/among_them/rag/seeder.py) would be the fix.

## File naming

Runners emit files with a stable prefix so the analyzer can bucket them:

- `exp_strat_<strategy>_<rep>.json` — qwen36 impostor vs qwen3-next crew, 5p/1i (legacy).
- `exp_gemma_strat_<strategy>_<rep>.json` — gemma impostor vs qwen36 crew, 5p/1i.
- `exp_big_a_<strategy>_<rep>.json` — 10p/2i/40r, gemma impostor, default crew.
- `exp_big_b_<strategy>_<rep>.json` — 10p/2i/40r, gemma impostor, reddit crew.
- `exp_big_rev_a_<strategy>_<rep>.json` — 10p/2i/40r, qwen36 impostor, default crew.
- `exp_big_rev_b_<strategy>_<rep>.json` — 10p/2i/40r, qwen36 impostor, reddit crew.

`_round_limit` suffix = game hit the round cap. `_exception` = the runner caught a fatal error.
