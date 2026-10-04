"""Populate the RAG store with curated strategies and parsed game logs.

Two entry points:

* ``seed_strategies(store)``    — upserts a fixed set of narrative strategies.
                                  Cheap, idempotent, safe to call on every boot.
* ``seed_from_game_logs(store)`` — parses every game log under ``RUNS_DIR``
                                  and indexes notable plays. Slow; run manually:
                                  ``python -m among_them.rag.seeder``.
"""

import json
import os
from typing import TYPE_CHECKING

from among_them.config import RUNS_DIR

if TYPE_CHECKING:
    from among_them.rag.store import RAGStore


# ---- Section A: curated strategies ------------------------------------

CURATED_STRATEGIES = [
    {
        "name": "afk_ghost",
        "role": "Impostor",
        "text": """AFK Ghost Strategy (Impostor): In the first action phase, do NOT kill
anyone. Instead pretend to do tasks and stay in Cafeteria or other busy rooms where
crewmates are visible. Do not engage aggressively. If anyone asks questions in
discussion, give short vague replies as if you were distracted or not paying attention
— act like someone who was AFK. This builds a low-suspicion, low-profile identity.
Starting from round 2 onwards, watch your player_in_room field carefully. The moment
you find yourself alone with exactly one crewmate and seen_actions shows no other
players nearby, execute the kill immediately. In the next discussion phase, continue
the AFK persona: say you just got back, express genuine confusion about the kill, and
subtly point suspicion at whoever reported the body by saying something like 'I only
just got back, I don't really know what happened, but [reporter name] was acting weird
earlier.' The AFK persona makes your lack of specific information completely plausible
and makes other players instinctively want to defend you.""",
    },
    {
        "name": "task_mimic",
        "role": "Impostor",
        "text": """Task Mimic Strategy (Impostor): Spend the first 2-3 action rounds
using PRETEND actions to fake completing tasks in high-traffic rooms like Cafeteria
and Medbay where multiple crewmates can witness you. This builds a strong visible
alibi. After establishing this reputation, move to an isolated room — Electrical,
Storage, or Navigation — and wait. Do not rush. If a crewmate follows you into the
isolated room and player_in_room shows only that one player, kill immediately. In
discussion, explicitly cite your earlier visible task activity: 'I was doing tasks in
Cafeteria the whole early game, everyone saw me.' Use Appeal to Logic framing to make
this sound like evidence rather than a claim.""",
    },
    {
        "name": "redirector",
        "role": "Impostor",
        "text": """Redirector Strategy (Impostor): When accused or questioned during
discussion, never directly deny the accusation — that sounds defensive. Instead
immediately redirect: pick another player, question their specific movements, and frame
your redirection as logical deduction rather than deflection. Example: 'I hear you but
actually — where was [target] during round 3? I saw them near the body location and
nobody's explained that yet.' Use Appeal to Logic to make your pivot sound analytical.
If any player even weakly corroborates your alibi, amplify it using Bandwagon Effect
framing: 'Even [name] agrees I was in Cafeteria.' Vote confidently for your redirect
target at the end.""",
    },
    {
        "name": "silent_hunter",
        "role": "Impostor",
        "text": """Silent Hunter Strategy (Impostor): Never kill in the first round.
Use the first round purely for intelligence — observe which crewmates are working
alone and which rooms are isolated. Track the seen_actions field: players who move
to isolated rooms like Electrical, Navigation, or Reactor frequently tend to be alone.
Kill only when seen_actions is empty (no witnesses) and player_in_room has exactly
one other player. After a kill, immediately move to a different room before the next
round so your last known location does not match the crime scene. In discussion use
Feigning Ignorance combined with Vagueness: 'I honestly don't know, I was moving
around doing tasks, I didn't see anything suspicious.'""",
    },
    {
        "name": "realistic_fake_tasks",
        "role": "Impostor",
        "text": """Realistic Fake Tasks Strategy (Impostor): When you choose a PRETEND
action at a task location, commit to it. Do NOT leave after a single round — real
tasks take time, and a player who arrives at Electrical, "completes" something
instantly, and immediately walks out reads as fake to anyone watching the round log.
Stay at the same task location for at least 2 consecutive action rounds doing the
same PRETEND action, then move on naturally as if the task is done. Avoid repeating
the exact same fake task across many separate visits to different rooms in the same
game — Task Verifier crewmates specifically look for impostors who claim the same
task multiple times because real tasks complete and disappear from your task list.
Mix your fake tasks across different rooms so your trail looks like a normal task
route, not a loop.""",
    },
    {
        "name": "post_kill_alibi",
        "role": "Impostor",
        "text": """Post-Kill Alibi Strategy (Impostor): The single most important move
after a kill is NOT the kill itself — it's where you are when the body is reported.
Execute the kill, then immediately in the next action round move OUT of the kill
room and into the highest-traffic room you can reach (Cafeteria, Medbay, or any
room your seen_actions shows multiple crewmates currently occupy). Pretend to do a
task there so other players witness your presence. When the body is reported in
discussion you can truthfully say "I was in Cafeteria with [name1] and [name2]"
and they will corroborate — two independent witnesses placing you elsewhere
collapses suspicion almost entirely. Never linger in or adjacent to the kill room
after the kill. If no high-traffic room is reachable, at least move TWO rooms away
so your last known location does not border the body.""",
    },
    {
        "name": "consistent_persona",
        "role": "Impostor",
        "text": """Consistent Persona Strategy (Impostor): Crewmates look for behavioural
changes between meetings. A player who is chatty and helpful early then suddenly
goes silent or terse the round a body is reported is flagged immediately. Decide on
your discussion persona in round 1 — length of messages, level of engagement,
willingness to accuse, willingness to defend others — and hold it constant for the
entire game regardless of whether you just killed someone. If you were a 2-sentence
contributor early, stay a 2-sentence contributor after the kill. If you asked one
clarifying question per discussion, keep asking one. Do not over-correct by suddenly
becoming the loudest voice in the room to "look innocent" — that is just as
suspicious as going silent. Match your own baseline.""",
    },
    {
        "name": "isolated_kill_rooms",
        "role": "Impostor",
        "text": """Isolated Kill Rooms Strategy (Impostor): Never kill in Cafeteria,
Hallway, or any room that crewmates routinely pass through — even if player_in_room
shows only one target, a third player can walk in mid-action and witness it. The
safe kill rooms are dead-ends and single-purpose rooms: Electrical, Navigation,
Reactor, and Communications. These rooms have low routine traffic and crewmates who
enter them are usually committed to a specific task there, meaning they will stay
put for a round or two — giving you a clean window. Before killing, check
seen_actions: if any other player moved INTO or PAST the kill room in the last 1-2
rounds, abort and reposition. Only execute when player_in_room has exactly one
target AND seen_actions is empty of any movement toward your room.""",
    },
    {
        "name": "patient_first_kill",
        "role": "Impostor",
        "text": """Patient First Kill Strategy (Impostor): The first kill round is the
most dangerous moment in the entire game — too early and the population is dense so
witnesses are likely; too late and your kill cooldown forces visibility you can't
afford. Wait at least until round 3-4. Use rounds 1-2 to fake tasks in high-traffic
rooms so multiple crewmates witness you doing "tasks" there — this builds the alibi
you'll cash in later. Only strike on round 3 or later when seen_actions has been
EMPTY for at least one consecutive round (meaning no one is moving near you) AND
player_in_room has exactly one target. If those conditions are not met, do not
attempt a kill that round — pretend tasks, wait one more round. Killing while any
witness movement is visible in seen_actions is the #1 way smaller-model impostors
lose. Patience > opportunism.""",
    },
    {
        "name": "second_kill_caution",
        "role": "Impostor",
        "text": """Second Kill Caution Strategy (Impostor): The second kill is what
gets impostors caught more often than the first. After your first kill succeeds and
the body is reported, crewmates have built witness chains and timelines — your
movement is now actively tracked. Do NOT kill in the round immediately after the
first body's discussion phase ends. Skip at least one full action round doing
visible PRETEND tasks in a busy room before attempting a second kill. When you do
take the second kill, pick a target who is NOT already in any witness's chain (i.e.
seen_actions shows them moving alone, unwitnessed, in recent rounds). Kill cooldown
permitting, this second-kill restraint reduces your detection rate dramatically.""",
    },
    {
        "name": "body_placement",
        "role": "Impostor",
        "text": """Body Placement Strategy (Impostor): Where you leave the body matters
as much as where you kill. After eliminating a target, the body remains in that
room until another player walks in and reports it. Killing in a peripheral room
like Reactor or Lower Engine means the body could go undiscovered for 2-3 rounds —
giving you extra action rounds (more potential kills) before the next discussion
phase. Killing in Medbay or Upper Engine means a crewmate doing tasks there
discovers the body in the next action round, forcing an immediate meeting that
costs you tempo. Prioritize kill locations that minimize the chance of immediate
body discovery: end-of-map rooms with no nearby tasks. Then move yourself OUT of
that arm of the map before crewmates start patrolling for the missing player.""",
    },
    {
        "name": "vote_split_engineering",
        "role": "Impostor",
        "text": """Vote Split Engineering Strategy (Impostor): You are outnumbered in
votes (1 vs many). You cannot win a vote by persuading the majority — you can only
win by making the majority disagree with each other. When discussion starts, do
not push one accusation hard; instead identify TWO crewmates whose stated locations
or movements partially contradict each other, and frame both as suspicious in
parallel: 'I'm torn — [name A] said they were in Cafeteria but I saw movement near
Medbay, and [name B] hasn't accounted for round 3 at all.' This forces crewmates
to vote on two different targets, splitting the meeting and dramatically lowering
the chance any single player including you gets enough votes to be banished. Even
if no one is voted out, you've burned a meeting without losing yourself — that's
a win for the impostor.""",
    },
    {
        "name": "preemptive_accusation",
        "role": "Impostor",
        "text": """Pre-emptive Accusation Strategy (Impostor): The single strongest
psychological signal a player is innocent is *being the first to accuse confidently*.
Real impostors usually wait to see which way the wind is blowing before speaking;
crewmates speak first because they have nothing to hide. Exploit this. The moment
the discussion phase opens after a body report, be the FIRST player to post a
concrete accusation with specifics — not vague suspicion. Cite a player's
seen_actions, a specific round number, and a specific room: 'In round 4 I saw
[target] move into Upper Engine alone right before the body was discovered there.'
Even if the cite is partially fabricated, it must sound forensically specific.
This shifts the burden of denial onto your target and frames YOU as the
investigator rather than the suspect.""",
    },
    {
        "name": "skip_round_camouflage",
        "role": "Impostor",
        "text": """Skip-Round Camouflage Strategy (Impostor): Do not kill every round
your cooldown permits. A pattern of "kill every available round" is detectable by
attentive crewmates — the body count grows on a regular cadence and the only
constant is your presence near each victim. Deliberately skip kill opportunities
1 in every 2-3 cooldown windows. On skip rounds, do PRETEND tasks in highly visible
rooms with crewmates watching. This breaks the kill cadence pattern and gives you
extra alibi witnesses for the rounds you DO kill. The cost (one missed kill) is
much smaller than the benefit (looking like a normal task-doing player on the
rounds it matters).""",
    },
    {
        "name": "witness_chain_break",
        "role": "Impostor",
        "text": """Witness Chain Break Strategy (Impostor): Crewmates win by building
"witness chains" — A saw B in Medbay, B saw C in Cafeteria, etc. — and identifying
the player who fits in no chain. If you find yourself listed in someone else's
seen_actions in a recent round, your priority next kill should be that witness if
their location is now isolated. Removing the witness erases the chain link that
ties YOU to a suspicious location. Conversely, never kill someone who has NOT
recently seen you — that target's death gives crewmates a fresh "where was X
killed" question that may put a spotlight on a room you were in. Pick targets
whose elimination either breaks a witness chain pointing at you, or whose absence
gives you no new investigative liability.""",
    },
]


CREWMATE_STRATEGIES = [
    {
        "name": "witness_chain",
        "role": "Crewmate",
        "text": """Witness Chain Strategy (Crewmate): Actively track which players you
have personally seen doing tasks or moving between specific rooms. In discussion,
construct a chain of verified sightings: 'I saw Alice in Medbay in round 1, Alice saw
Bob in Cafeteria, Bob can vouch for Charlie.' Any player who cannot be placed in any
witness chain across any round is your highest priority suspect. Use Shifting the
Burden of Proof to demand their specific locations: 'Tell us exactly where you were in
rounds 1 and 2 and who can confirm it.' Vague or contradictory answers are strong
impostor tells — press immediately.""",
    },
    {
        "name": "early_reporter",
        "role": "Crewmate",
        "text": """Early Reporter Strategy (Crewmate): Report dead bodies immediately.
The moment you find a body, call it. In discussion be extremely specific about what
your seen_actions log shows — quote it directly. The player who last moved into or
out of the body's room in your seen_actions is the highest priority suspect. Ask
other players to corroborate or contradict that movement. If someone's stated location
contradicts what your seen_actions shows, call it out using Appeal to Logic: 'You said
you were in Storage but I saw you move to Electrical right before the body was found
— that doesn't match.'""",
    },
    {
        "name": "task_verifier",
        "role": "Crewmate",
        "text": """Task Verifier Strategy (Crewmate): Focus on completing your own
tasks as fast as possible — crewmates win by task completion. Move efficiently
between your task locations without unnecessary detours. In discussion, ask other
players to name the specific tasks they completed and where. Impostors using PRETEND
actions often repeat the same fake task (e.g. always 'Start the coffee maker in
Cafeteria') because it's available every round. If a player claims the same task
multiple times across rounds, flag this: real tasks complete and disappear, fake
tasks can be pretended indefinitely.""",
    },
    {
        "name": "buddy_system",
        "role": "Crewmate",
        "text": """Buddy System Strategy (Crewmate): Reddit consensus — never travel
alone in the early-to-mid game. When choosing your next move, prefer rooms that
player_in_room shows already have at least one other crewmate, even if your task
is elsewhere. An impostor cannot risk killing you with a witness present. Rotate
buddies between rounds — staying paired with the same player every round means
that if THEY are the impostor, you've been their alibi the whole game. In
discussion, explicitly state who you were paired with each round and ask them to
confirm — mutual corroboration is the strongest crewmate signal. If a player
refuses to name who they were with, treat that as a tell.""",
    },
    {
        "name": "skeptical_voter",
        "role": "Crewmate",
        "text": """Skeptical Voter Strategy (Crewmate): Reddit's most repeated advice
— do NOT vote in the first meeting unless a player's stated location directly
contradicts what your seen_actions log shows. Impostors win meetings by pushing
the crew to vote out an innocent player on weak evidence. Default to 'vote for
nobody' unless you have a hard contradiction. Watch for impostor tells: confident
accusation with no specifics, immediate redirection when questioned, pushing the
group toward a quick vote ('we have to vote SOMEONE'). Skipping a meeting costs
nothing; voting out a crewmate hands the impostor the game.""",
    },
    {
        "name": "behavioral_profiler",
        "role": "Crewmate",
        "text": """Behavioral Profiler Strategy (Crewmate): Reddit pro tip — track
how each player BEHAVES across meetings, not just where they were. Note who
spoke first in meeting 1, how long their messages were, who they accused, who
they defended. In meeting 2, compare. A player who went from chatty-defensive
to silent-and-terse the round a body dropped is the textbook post-kill
behavioural shift impostors make to 'play it cool'. A player who suddenly
becomes the loudest accuser after staying quiet for two rounds is over-correcting.
In discussion, cite the shift explicitly: 'In round 2 you had three points to
make, in round 4 you've said one thing. What changed?' Use Appeal to Logic
framing.""",
    },
]


def seed_strategies(store: "RAGStore") -> None:
    """Upsert all curated strategies into ``store.strategies``.

    Roles are normalized to lowercase to match what the retriever queries
    with (``store.query_strategy(..., role="impostor")``).
    """
    for s in CURATED_STRATEGIES + CREWMATE_STRATEGIES:
        store.add_strategy(
            text=s["text"],
            role=s["role"].lower(),
            strategy_name=s["name"],
        )


# ---- Section B: parse past games --------------------------------------

def _classify_outcome(playthrough: list, file_name: str) -> "str | None":
    """Return ``'crew'``, ``'imp'``, or ``None`` for a draw / truncated game.

    We require an explicit positive marker so an aborted game (no end-state
    line) cannot silently be miscredited to one side. ``_round_limit`` files
    deliberately produce no winner marker; we also catch any other truncation
    the same way.
    """
    has_crew_win = any("Crewmates win!" in line for line in playthrough)
    has_imp_win = any("Impostors win!" in line for line in playthrough)
    if has_crew_win and not has_imp_win:
        return "crew"
    if has_imp_win and not has_crew_win:
        return "imp"
    # No marker, both markers (shouldn't happen), or explicit round-limit
    # filename — outcome is ambiguous, drop the game from the index.
    return None


def seed_from_game_logs(
    store: "RAGStore", tournament_dir: str = RUNS_DIR
) -> None:
    """Scan tournament logs and index notable action/discussion plays.

    For each *decisive* game (one with a single positive winner marker) we keep:

    * every kill action (regardless of outcome — losing kills are still
      informative for the opposing side)
    * every other action by a player on the winning side
    * every discussion statement made by the player, tagged with that
      player's win/loss outcome

    Games with no winner marker (round-limit draws or truncated saves) are
    skipped — labelling them as either side's win would poison retrieval.
    Within a single game, identical chat statements (which accumulate across
    rounds because ``chat_messages`` is a growing log) are de-duplicated so
    we don't pay for the same embedding several times.

    Errors are logged and skipped so a malformed file can't stop the seed.
    """
    if not os.path.isdir(tournament_dir):
        print(f"⚠️  Tournament dir not found: {tournament_dir}")
        return

    files = sorted(f for f in os.listdir(tournament_dir) if f.endswith(".json"))
    n_total = len(files)
    n_indexed = n_skipped_draw = n_skipped_err = 0
    n_actions = n_discussion = 0

    for idx, file_name in enumerate(files, 1):
        try:
            with open(os.path.join(tournament_dir, file_name)) as f:
                data = json.load(f)

            outcome = _classify_outcome(data.get("playthrough", []), file_name)
            if outcome is None:
                n_skipped_draw += 1
                continue
            crew_won = outcome == "crew"

            for player in data["players"]:
                role = player["role"]  # "Impostor" or "Crewmate"
                role_lower = role.lower()
                won = (not crew_won) if role == "Impostor" else crew_won

                rounds = player.get("history", {}).get("rounds", [])
                # Per-game dedup set for chat messages — chat_messages
                # accumulates across rounds, so the same statement appears
                # in every later round's chat_messages list.
                seen_statements: set = set()

                for r in rounds:
                    stage = r.get("stage", "")

                    if stage == "Action Phase":
                        action_result = r.get("action_result", "")
                        if not action_result:
                            continue

                        llm_responses = r.get("llm_responses", [])
                        action = (
                            llm_responses[1]
                            if len(llm_responses) > 1
                            else r.get("response", "")
                        )
                        seen = r.get("seen_actions", [])

                        situation = (
                            f"Role: {role} | "
                            f"Location: {r.get('location', '')} | "
                            f"In room: {r.get('player_in_room', '')} | "
                            f"Recently seen: "
                            f"{'; '.join(seen[-2:]) if seen else 'nothing'}"
                        )

                        if "eliminated" in action_result or won:
                            store.add_action_play(
                                situation=situation,
                                action=action,
                                outcome=action_result,
                                role=role_lower,
                                won=won,
                                location=r.get("location", ""),
                                source_file=file_name,
                            )
                            n_actions += 1

                    elif stage == "Discuss":
                        chat_messages = r.get("chat_messages", [])
                        player_name = player["name"]
                        name_tag = f"[{player_name}]:"

                        # ``chat_messages`` mixes:
                        #   * player utterances ('Discussion: [Bob]: ...')
                        #   * the same utterance once history starts
                        #     accumulating ('chat: Discussion: [Bob]: ...')
                        #   * System notifications ('Discussion: [System]:
                        #     You have N rounds left to discuss...')
                        # Strip the 'chat:' prefix so an utterance has a
                        # stable canonical form, and drop System lines —
                        # both are pure noise that bloats the embedding.
                        cleaned: list = []
                        for m in chat_messages:
                            if not m or "[System]:" in m:
                                continue
                            cleaned.append(
                                m[6:] if m.startswith("chat: ") else m
                            )

                        context = "\n".join(cleaned[-3:]) if cleaned else ""

                        for msg in cleaned:
                            if name_tag not in msg:
                                continue
                            if msg in seen_statements:
                                continue
                            seen_statements.add(msg)
                            store.add_discussion_play(
                                situation=context,
                                statement=msg,
                                techniques=[],
                                role=role_lower,
                                won=won,
                                source_file=file_name,
                            )
                            n_discussion += 1
            n_indexed += 1
        except Exception as e:
            n_skipped_err += 1
            print(f"⚠️  Skipped {file_name}: {e}")

        # Compact progress line — overwrites the same row.
        if idx % 25 == 0 or idx == n_total:
            print(
                f"\r[{idx:>4}/{n_total}] indexed={n_indexed} "
                f"draws={n_skipped_draw} errors={n_skipped_err} "
                f"actions+={n_actions} discussion+={n_discussion}",
                end="",
                flush=True,
            )
    print()  # finish the progress line


if __name__ == "__main__":
    from among_them.rag.store import RAGStore  # noqa: WPS433 (local import OK in CLI)

    store = RAGStore()
    print("Seeding curated strategies...")
    seed_strategies(store)
    print("Parsing game logs...")
    seed_from_game_logs(store)
    print(
        f"Done. Actions: {store.action_plays.count()}, "
        f"Discussion: {store.discussion_plays.count()}, "
        f"Strategies: {store.strategies.count()}"
    )
