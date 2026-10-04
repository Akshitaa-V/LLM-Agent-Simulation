"""Shared metadata for the 10p/2i/40-round big/reverse batch experiments.

Both the Streamlit ablation view (``gui_handler.py``) and the LLM-judge
script (``scripts/judge_games.py``) need to map a run filename like
``exp_big_rev_a_post_kill_alibi_3.json`` to its batch tag (``rev_a``) and
strategy name (``post_kill_alibi``). Keeping one definition here means the
consumers can't drift out of sync with each other.
"""

BATCH_PREFIX_TAGS = {
    "exp_big_rev_a_": "rev_a",
    "exp_big_rev_b_": "rev_b",
    "exp_big_a_": "big_a",
    "exp_big_b_": "big_b",
}

SETUP_LABELS = {
    "big_a": "gemma4 imp | default crew",
    "big_b": "gemma4 imp | reddit crew",
    "rev_a": "qwen36 imp | default crew",
    "rev_b": "qwen36 imp | reddit crew",
}


def batch_tag(filename: str) -> str | None:
    """Return the batch tag (e.g. ``"big_a"``) for a run filename, or None."""
    for prefix, tag in BATCH_PREFIX_TAGS.items():
        if filename.startswith(prefix):
            return tag
    return None


def strategy_name(filename: str) -> str:
    """Strip batch prefix, extension, retry suffix, and trailing seed index."""
    body = filename
    for prefix in BATCH_PREFIX_TAGS:
        if body.startswith(prefix):
            body = body[len(prefix):]
            break
    body = body.removesuffix(".json")
    for suffix in ("_round_limit", "_exception"):
        if body.endswith(suffix):
            body = body[: -len(suffix)]
    parts = body.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else body
