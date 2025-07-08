from langgraph.graph import END, StateGraph
import streamlit as st
from typing_extensions import TypedDict
from typing import List, TypedDict

### Helper functions for the notebook
from nodes.nodes import planner, process_memory, react_agent

openai_api_key = st.secrets["OPENAI_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]
    
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

def create_agent():
    agent_workflow = StateGraph(PlanExecute)

    # Add the anonymize node
    agent_workflow.add_node("process_memory", process_memory)
    # agent_workflow.add_node("planner", planner)
    agent_workflow.add_node("react_agent", react_agent)
    # Set the entry point
    agent_workflow.set_entry_point("process_memory")
    agent_workflow.add_edge("process_memory", "react_agent")
    agent_workflow.add_edge("react_agent", END)

    plan_and_execute_app = agent_workflow.compile()

    return plan_and_execute_app