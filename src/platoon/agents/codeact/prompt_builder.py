from pathlib import Path
from platoon.envs.codeact import CodeActObservation, CodeActAction
from platoon.utils.llm_client import ChatMessage, Conversation, ConversationWithMetadata
from platoon.utils.prompt_retriever import PromptRetriever


class CodeActPromptBuilder:

    def __init__(self, prompts_dir: str | Path | None = None):
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent / "prompts"
        self.retriever = PromptRetriever(prompts_dir=prompts_dir)
        
    # TODO: We need to refactor this to be more general.
    def build_messages_from_traj_dump(self, traj_collection_dump: dict, reward_threshold: float) -> list[ConversationWithMetadata]:
        raise NotImplementedError("Subclasses must implement this method")

            
    # TODO: Revisit if we want to separate into prompt and completion keys.
    def build_messages(self, obs: CodeActObservation, agent_action: CodeActAction | None = None) -> Conversation:
        system_prompt = self.build_system_prompt(obs)
        user_prompt = self.build_user_prompt(
            obs,
            task=str(obs.task),
            action_space_description=str(obs.action_space),
            action_history_description=self.build_action_history_description(obs),
            next_action_str=self.build_next_action_str(obs),
        )

        messages: Conversation = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        if agent_action:
            messages.append(ChatMessage(role="assistant", content=str(agent_action)))
        
        return messages
    
    # TODO: Support history truncation in the future.
    def build_action_history_description(self, obs: CodeActObservation) -> str:
        if not obs.history:
            return "No actions taken yet."

        processed_cells = [f"Cell {i}:\n" + str(cell) for i, cell in enumerate(obs.history)]
        return "\n".join(processed_cells)
    
    # TODO: Revisit whether we want to use obs inside or pass from outside.
    def build_next_action_str(self, obs: CodeActObservation, **context) -> str:
        return self.retriever.get_prompt("user-next-action-str", **context)

    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        return self.retriever.get_prompt("system", **context)
    
    def build_user_prompt(self, obs: CodeActObservation, **context) -> str:
        return self.retriever.get_prompt("user", **context)
