from langchain.agents.agent import AgentAction
import re
from typing import Dict, List, Optional

def get_last_tool_input(intermediate_steps, tool_name="interact_with_human"):
    for step in reversed(intermediate_steps):
        action = step if isinstance(step, AgentAction) else step[0]
        if action.tool == tool_name:
            return action.tool_input
    return None

def get_last_log(intermediate_steps, tool_name=None):
    for step in reversed(intermediate_steps):
        action = step if isinstance(step, AgentAction) else step[0]
        if tool_name is None or action.tool == tool_name:
            return action.log
    return None

def insert_observation_for_last_interact_human(intermediate_steps, observation):
    """
    Insert observation for the last interact_with_human tool in intermediate steps
    
    Args:
        intermediate_steps: List of AgentAction or (AgentAction, observation) tuples
        observation: String observation to insert
    
    Returns:
        Updated intermediate_steps with observation inserted
    """
    if not intermediate_steps:
        return intermediate_steps
    
    # Work backwards to find the last interact_with_human
    for i in range(len(intermediate_steps) - 1, -1, -1):
        step = intermediate_steps[i]
        
        # Case 1: step is a tuple (AgentAction, observation)
        if isinstance(step, tuple) and len(step) == 2:
            action, existing_observation = step
            if hasattr(action, 'tool') and action.tool == 'interact_with_human':
                # Update the observation
                intermediate_steps[i] = (action, f"User menjawab: {observation}")
                return intermediate_steps
        
        # Case 2: step is just an AgentAction (no observation yet)
        elif hasattr(step, 'tool') and step.tool == 'interact_with_human':
            # Convert to tuple with observation
            intermediate_steps[i] = (step, f"User menjawab: {observation}")
            return intermediate_steps
    
    return intermediate_steps

def clean_agent_log(log: str) -> str:
    log = re.sub(r'^Thought:\s*', '', log, flags=re.MULTILINE)
    log = re.sub(r'Action:.*?(?=^Thought:|$)', '', log, flags=re.DOTALL | re.MULTILINE)
    log = re.sub(r'Action Input:.*?(?=^Thought:|$)', '', log, flags=re.DOTALL | re.MULTILINE)
    log = re.sub(r'\n\s*\n', '\n\n', log).strip()
    return log

def format_scratchpad_from_steps(intermediate_steps):
    """Convert previous steps to ReAct scratchpad format string."""
    scratchpad = ""
    for action, observation in intermediate_steps:
        scratchpad += f"{action.log.strip()}\n"
        scratchpad += f"Observation: {observation}\n"
    return scratchpad

def get_last_user_message(chat_history: List[Dict[str, str]]) -> Optional[str]:
    for message in reversed(chat_history):
        if message.get("role") == "user":
            return message.get("content")
    return None  # No user message found