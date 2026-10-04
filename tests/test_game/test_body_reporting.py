from typing import List

from pydantic import Field

from among_them.game.game_engine import GameEngine
from among_them.game.models.engine import GameLocation, GamePhase
from among_them.game.models.history import PlayerState
from among_them.game.players.base_player import Player, PlayerRole
from among_them.game.players.human import HumanPlayer


class ScriptedPlayer(HumanPlayer):
    """Human player with deterministic action choices for engine tests."""

    scripted_action_indices: List[int] = Field(default_factory=list)
    script_cursor: int = 0

    def prompt_action(self, actions: List[str], extra_context: str = "") -> int:
        if self.script_cursor >= len(self.scripted_action_indices):
            raise IndexError(
                f"No scripted action left for {self.name}; got options: {actions}"
            )
        action_index = self.scripted_action_indices[self.script_cursor]
        self.script_cursor += 1
        self.state.actions = actions
        self.state.response = str(action_index)
        return action_index

    def prompt_discussion(self) -> str:
        return "test discussion"

    def prompt_vote(self, voting_actions: List[str], dead_players: List[str]) -> int:
        return 0


def _setup_engine(
    players: List[Player], *, player_to_act_next: int = 0, round_number: int = 0
) -> GameEngine:
    engine = GameEngine()
    engine.state.players = players
    engine.state.game_stage = GamePhase.ACTION_PHASE
    engine.state.player_to_act_next = player_to_act_next
    engine.state.round_number = round_number
    return engine


def test_move_into_body_offers_immediate_report():
    discoverer = ScriptedPlayer(
        name="Doraemon",
        scripted_action_indices=[1, 1],
    )
    discoverer.state.location = GameLocation.LOC_UPPER_ENGINE

    victim = HumanPlayer(name="Macroon")
    victim.state.life = PlayerState.DEAD
    victim.state.location = GameLocation.LOC_MEDBAY

    bystander = HumanPlayer(name="Dave")

    engine = _setup_engine([discoverer, victim, bystander])
    move_to_medbay_index = engine.get_actions(discoverer).index(
        next(
            action
            for action in engine.get_actions(discoverer)
            if action.target == GameLocation.LOC_MEDBAY
        )
    )
    discoverer.scripted_action_indices = [move_to_medbay_index, 1]

    engine.perform_action_step()

    assert discoverer.state.location == GameLocation.LOC_MEDBAY
    assert engine.state.game_stage == GamePhase.DISCUSS
    assert victim.state.life == PlayerState.DEAD_REPORTED
    assert any("body_discovered" in entry for entry in engine.state.playthrough)
    assert any("report: Doraemon reported a dead body" in entry for entry in engine.state.playthrough)
    assert not any("body_skipped" in entry for entry in engine.state.playthrough)


def test_move_into_body_can_be_skipped():
    discoverer = ScriptedPlayer(
        name="Doraemon",
        scripted_action_indices=[1, 0],
    )
    discoverer.state.location = GameLocation.LOC_UPPER_ENGINE

    victim = HumanPlayer(name="Macroon")
    victim.state.life = PlayerState.DEAD
    victim.state.location = GameLocation.LOC_MEDBAY

    bystander = HumanPlayer(name="Dave")

    engine = _setup_engine([discoverer, victim, bystander])
    move_to_medbay_index = engine.get_actions(discoverer).index(
        next(
            action
            for action in engine.get_actions(discoverer)
            if action.target == GameLocation.LOC_MEDBAY
        )
    )
    discoverer.scripted_action_indices = [move_to_medbay_index, 0]

    engine.perform_action_step()

    assert engine.state.game_stage == GamePhase.ACTION_PHASE
    assert victim.state.life == PlayerState.DEAD
    assert any("body_skipped" in entry for entry in engine.state.playthrough)
    assert any("(found dead body of Macroon)" in entry for entry in engine.state.playthrough)


def test_kill_does_not_offer_report_followup_to_impostor():
    impostor = ScriptedPlayer(
        name="Modi",
        role=PlayerRole.IMPOSTOR,
        scripted_action_indices=[0],
    )
    impostor.kill_cooldown = 0
    impostor.state.location = GameLocation.LOC_MEDBAY

    victim = HumanPlayer(name="Macroon")
    victim.state.location = GameLocation.LOC_MEDBAY

    bystander = HumanPlayer(name="Dave")

    engine = _setup_engine([impostor, victim, bystander])
    kill_index = engine.get_actions(impostor).index(
        next(
            action
            for action in engine.get_actions(impostor)
            if action.target == victim
        )
    )
    impostor.scripted_action_indices = [kill_index]

    engine.perform_action_step()

    assert victim.state.life == PlayerState.DEAD
    assert engine.state.game_stage == GamePhase.ACTION_PHASE
    assert any("(left dead body of Macroon)" in entry for entry in engine.state.playthrough)
    assert any(
        "body_discovered: You eliminated Macroon" in obs
        for obs in impostor.state.observations
    )
    assert not any("body_skipped" in entry for entry in engine.state.playthrough)


def test_unreported_body_stays_discoverable_after_voting():
    reporter = ScriptedPlayer(name="Reporter", scripted_action_indices=[0])
    reporter.state.location = GameLocation.LOC_MEDBAY

    reported_victim = HumanPlayer(name="Macroon")
    reported_victim.state.life = PlayerState.DEAD
    reported_victim.state.location = GameLocation.LOC_MEDBAY

    hidden_victim = HumanPlayer(name="Charlie")
    hidden_victim.state.life = PlayerState.DEAD
    hidden_victim.state.location = GameLocation.LOC_CAFETERIA

    voter = ScriptedPlayer(name="Voter", scripted_action_indices=[0])

    engine = _setup_engine([reporter, reported_victim, hidden_victim, voter])
    report_index = engine.get_actions(reporter).index(
        next(
            action
            for action in engine.get_actions(reporter)
            if "report dead body" in action.text
        )
    )
    reporter.scripted_action_indices = [report_index]

    engine.perform_action_step()

    assert reported_victim.state.life == PlayerState.DEAD_REPORTED
    assert hidden_victim.state.life == PlayerState.DEAD

    engine.go_to_voting()

    assert reported_victim.state.life == PlayerState.DEAD_REPORTED
    assert hidden_victim.state.life == PlayerState.DEAD
