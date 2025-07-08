
from langchain.agents.agent import AgentAction
from typing import Any, List, Tuple
from langchain.callbacks.base import BaseCallbackHandler


class StopOnToolCallback(BaseCallbackHandler):
    def __init__(self, stop_on_tool: str):
        self.stop_on_tool = stop_on_tool
        self.intermediate_steps: List[Tuple[AgentAction, str]] = []
        self.pending_action: AgentAction | None = None

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        print(f"[Callback] Agent chose tool: {action.tool}")
        self._latest_action = action
        print("ACTION: ", action)
        self.intermediate_steps.append((self._latest_action))

        # Stop only on the specified tool
        if self._latest_action.tool == self.stop_on_tool:
            print(f"[Stop] Tool '{self.stop_on_tool}' called with input: {self._latest_action.tool_input}")
            raise KeyboardInterrupt("Stopped after specified tool was used.")