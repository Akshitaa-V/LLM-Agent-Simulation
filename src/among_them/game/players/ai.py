from typing import List

from among_them.game.agents.adventure_agent import AdventureAgent
from among_them.game.agents.discussion_agent import DiscussionAgent
from among_them.game.agents.voting_agent import VotingAgent
from among_them.game.consts import TOKEN_COSTS
from among_them.game.models.usage_metadata import UsageMetadata
from among_them.game.players.base_player import Player


class AIPlayer(Player):
    llm_model_name: str
    # When True, the three sub-agents (adventure / discussion / voting) will
    # query the RAG store and inject retrieved strategies + past winning plays
    # into their prompts. Defaults to False so existing games are unaffected.
    use_rag: bool = False
    # Per-strategy ablation: when this is non-empty, the same strategy text
    # is injected verbatim into every sub-agent's prompt for the whole game,
    # bypassing the RAG store entirely. Takes precedence over ``use_rag``.
    injected_strategy: str = ""

    def __init__(self, **data):
        super().__init__(**data)  # Initialize Player fields first
        self.adventure_agent = AdventureAgent(
            llm_model_name=self.llm_model_name,
            player_name=self.name,
            role=self.role.value,
            use_rag=self.use_rag,
            injected_strategy=self.injected_strategy,
        )
        self.discussion_agent = DiscussionAgent(
            llm_model_name=self.llm_model_name,
            player_name=self.name,
            role=self.role.value,
            use_rag=self.use_rag,
            injected_strategy=self.injected_strategy,
        )
        self.voting_agent = VotingAgent(
            llm_model_name=self.llm_model_name,
            player_name=self.name,
            role=self.role.value,
            use_rag=self.use_rag,
            injected_strategy=self.injected_strategy,
        )

    def prompt_action(self, actions: List[str], extra_context: str = "") -> int:
        self.state.actions = actions
        history = self.history.get_history_str()
        teammate_block = self._teammate_block()
        if teammate_block:
            history = teammate_block + history
        if extra_context:
            history = extra_context + "\n\n" + history
        prompts, chosen_action = self.adventure_agent.act(
            observations=history,
            tasks=self.get_task_to_complete(),
            actions=actions,
            current_location=self.state.location.value,
            in_room=self.state.player_in_room,
        )
        self.state.llm_responses = self.adventure_agent.responses
        self.add_token_usage(self.adventure_agent.token_usage)
        self.state.response = str(chosen_action)
        self.state.prompts = prompts
        return chosen_action

    def prompt_discussion(self) -> str:
        history = self.history.get_history_str()
        teammate_block = self._teammate_block()
        if teammate_block:
            history = teammate_block + history
        statements = "\n".join(self.get_chat_messages())
        message_prompt, message = self.discussion_agent.act(
            observations=history, messages=statements
        )
        self.state.llm_responses = self.discussion_agent.responses
        self.add_token_usage(self.discussion_agent.token_usage)
        self.state.response = message
        self.state.prompts = message_prompt
        return message

    def prompt_vote(self, voting_actions: List[str], dead_players: List[str]) -> int:
        self.state.actions = voting_actions
        history = self.history.get_history_str()
        teammate_block = self._teammate_block()
        if teammate_block:
            history = teammate_block + history
        vote_prompt, vote = self.voting_agent.act(
            observations=history,
            actions=voting_actions,
            discussion_log="\n".join(self.get_chat_messages()),
            dead_players=dead_players,
        )
        self.state.llm_responses = self.voting_agent.responses
        self.add_token_usage(self.voting_agent.token_usage)
        self.state.response = str(vote)
        self.state.prompts = vote_prompt
        return vote

    def _teammate_block(self) -> str:
        if not self.is_impostor or not self.fellow_impostors:
            return ""
        names = ", ".join(self.fellow_impostors)
        return (
            f"=== TEAMMATE INFO ===\n"
            f"You are an impostor. Your fellow impostor(s): {names}.\n"
            f"Do NOT eliminate them, do NOT vote them out, and coordinate "
            f"with them in discussions to deflect suspicion.\n"
            f"=====================\n\n"
        )

    def add_token_usage(self, usage: UsageMetadata):
        self.state.token_usage.input_tokens += usage.input_tokens
        self.state.token_usage.output_tokens += usage.output_tokens
        self.state.token_usage.total_tokens += usage.total_tokens
        self.state.token_usage.cache_read += usage.cache_read

        for_model = self.llm_model_name
        previous_cost = self.state.token_usage.cost
        self.state.token_usage.cost = 0

        # if ends with :free, cost is 0
        if for_model.endswith(":free"):
            return
        if for_model not in TOKEN_COSTS:
            print(
                f"Model {for_model} not found in TOKEN_COSTS. "
                "Defaulting to openai/gpt-4o-mini"
            )
            for_model = "openai/gpt-4o-mini"
        million = 1_000_000
        self.state.token_usage.cost += (
            self.state.token_usage.input_tokens
            * TOKEN_COSTS[for_model]["input_tokens"]
            / million
        )
        self.state.token_usage.cost += (
            self.state.token_usage.output_tokens
            * TOKEN_COSTS[for_model]["output_tokens"]
            / million
        )
        self.state.token_usage.cost += (
            self.state.token_usage.cache_read
            * TOKEN_COSTS[for_model]["cache_read"]
            / million
        )
        current_cost = round(self.state.token_usage.cost - previous_cost, 6)
        total_cost = round(self.state.token_usage.cost, 6)
        print(
            f"\033[90m Player cost (action/total): {current_cost}/{total_cost} \033[00m"
        )

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
