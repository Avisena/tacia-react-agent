from pprint import pprint
from typing_extensions import TypedDict
from langchain.agents.agent import AgentAction, AgentFinish
from typing import List, TypedDict
from helpers.helpers import get_last_tool_input, insert_observation_for_last_interact_human, clean_agent_log, get_last_user_message, get_last_log
from chains.chains import process_memory_chain, planner_chain, create_react_agent_chain, self_reflection_chain, semantic_summary_chain
from callbacks.callbacks import StopOnToolCallback
from tools.tools import ask_ai, interact_human, search_web


class PlanExecute(TypedDict):
    curr_state: str
    question: str
    memory_based_question: str
    intermediate_steps: List
    chat_history: List
    anonymized_question: str
    query_to_retrieve_or_answer: str
    plan: List[str]
    past_steps: List[str]
    mapping: dict 
    curr_context: str
    aggregated_context: str
    tool: str
    response: str
    

def planner(state:PlanExecute):
    """
    Plans the next step.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the plan.
    """
    state["curr_state"] = "planner"
    print("Planning step")
    pprint("--------------------")
    plan = planner_chain.invoke({"memory_based_question": state['memory_based_question']})
    state["plan"] = plan.steps
    print(f'plan: {state["plan"]}')
    return state

def process_memory(state: PlanExecute):
    """
    process the memory to make question.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the anonymized question and mapping.
    """
    print("Anonymizing question")
    pprint("--------------------")
    input_values = {"chat_history": state['chat_history']}
    print("CURR_STATE: ", state['curr_state'])
    state["curr_state"] = "process_memory"
    process_memory_output = process_memory_chain.invoke(input_values)
    memory_based_question = process_memory_output["memory_based_question"]
    print(f'memory_based_question: {memory_based_question}')
    state['memory_based_question'] = memory_based_question
    return state


def react_agent(state:PlanExecute):
    """
    Plans using react agent
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the plan.
    """
    callback_handler = StopOnToolCallback(stop_on_tool="interact_human")
    react_agent_chain = create_react_agent_chain(callback_handler)
    state["curr_state"] = "processing_react_agent"
    print("React agent step")
    pprint("--------------------")
    intermediate_steps = state['intermediate_steps']
    # scratchpad = format_scratchpad_from_steps(intermediate_steps)
    print("state['intermediate_steps']: ", state['intermediate_steps'])
    try:
        next_input = {
        "input": state['memory_based_question'],
        }
        if len(state['intermediate_steps']) == 0:
            print("ENTERING 0 INTERMEDIATE STEPS")
            answer = react_agent_chain.invoke(next_input)
        else:
            answer = continue_agent_reasoning(react_agent_chain, state['memory_based_question'], state['intermediate_steps'], callback_handler)
            print("ANSWER: ", answer)
            state["chat_history"].append({"role": "assistant_reasoning", "content": clean_agent_log(answer['log'])})
        state['response'] = answer['output']
        state["curr_state"] = "finish_react_agent"
        state["intermediate_steps"] = []
    except KeyboardInterrupt:
        print("Agent stopped after hitting tool:", callback_handler.stop_on_tool)
        print(callback_handler.intermediate_steps)
        state['intermediate_steps'].extend(callback_handler.intermediate_steps)
        print(f"Total saved steps length: {len(state['intermediate_steps'])}")
        print(f"Total saved steps: {state['intermediate_steps']}")
        last_log = get_last_log(callback_handler.intermediate_steps)
        state["chat_history"].append({"role": "assistant_reasoning", "content": clean_agent_log(last_log)})
        state['chat_history'].append({"role": "assistant", "content": get_last_tool_input(callback_handler.intermediate_steps)})
    return state

def self_reflection(state: PlanExecute):
    state["curr_state"] = "self_reflection"
    print("self_reflection step")
    pprint("--------------------")
    self_reflection = self_reflection_chain.invoke({"question": state['memory_based_question'], "answer": state["response"]})
    state["response"] = self_reflection["improved_answer"]
    state["chat_history"].append({"role": "assistant", "content": state["response"]})
    return state


def semantic_summary(state: PlanExecute):
    print("semantic summary step")
    pprint("--------------------")
    print(">>> RAW STATE:", state)
    print(">>> TYPE OF STATE:", type(state))
    # # print(f"MEMORY BASED QUESTION: {state["memory_based_question"]}")
    # semantic_summary = semantic_summary_chain.invoke({"user_input": state['memory_based_question']})
    # print("Semantic Summary: ", semantic_summary['content'])

def is_self_reflection(state: PlanExecute):
    label = state.get("curr_state")
    if label == "finish_react_agent":
        return "finish_react_agent"
    
    else:
        return "not_finish_react_agent"

def is_processing_react_agent(state: PlanExecute):
    label = state.get("curr_state")
    if label == "processing_react_agent":
        print("INTERMEDIATE_STEPS: ", state['intermediate_steps'])
        print("CHAT_HISTORY: ", state['chat_history'])
        print(get_last_user_message(state['chat_history']))
        state['intermediate_steps'] = insert_observation_for_last_interact_human(state['intermediate_steps'], get_last_user_message(state['chat_history']))
        print("INSERT ANSWER TO INT STEPS: ", state['intermediate_steps'])
        return "processing_react_agent"
    else:
        return "not_processing_react_agent"

def continue_agent_reasoning(agent_executor, input_text, intermediate_steps, callback_handler=None):
    """Continue agent reasoning from intermediate steps with callback support"""
    
    current_steps = list(intermediate_steps)
    max_iterations = 10
    
    for i in range(max_iterations):
        # Get the agent's next action
        next_step = agent_executor.agent.plan(
            intermediate_steps=current_steps,
            input=input_text
        )
        
        if isinstance(next_step, AgentFinish):
            last_log = current_steps[-1][0].log if current_steps else ""
            return {
                "input": input_text,
                "output": next_step.return_values.get("output", ""),
                "intermediate_steps": intermediate_steps,
                "log": last_log
            }
        
        # Execute the action
        if isinstance(next_step, AgentAction):
            # Manually trigger the callback if provided
            if callback_handler:
                callback_handler.on_agent_action(next_step)
            
            tool_name = next_step.tool
            tool_input = next_step.tool_input
            
            # Find and execute the tool
            tool = next((t for t in agent_executor.tools if t.name == tool_name), None)
            if tool:
                observation = tool.run(tool_input)
                
                # Manually trigger tool end callback if provided
                # if callback_handler:
                #     callback_handler.on_tool_end(observation)
                
                # Add to intermediate steps
                current_steps.append((next_step, observation))
            else:
                observation = f"Tool {tool_name} not found"
                current_steps.append((next_step, observation))
    
    return {"output": "Max iterations reached", "intermediate_steps": current_steps}