from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableLambda
import streamlit as st
from typing_extensions import TypedDict
from typing import List, TypedDict

### Helper functions for the notebook
from nodes.nodes import planner, process_memory, react_agent, self_reflection, is_self_reflection, is_processing_react_agent

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

    # agent_workflow.add_node("planner", planner)
    agent_workflow.add_node("start", RunnableLambda(lambda state: state))
    agent_workflow.add_node("process_memory", process_memory)
    agent_workflow.add_node("react_agent", react_agent)
    agent_workflow.add_node("self_reflection", self_reflection)

    agent_workflow.set_entry_point("start")
    agent_workflow.add_conditional_edges(
    "start",
    is_processing_react_agent,
        {
            "processing_react_agent": "react_agent",
            "not_processing_react_agent": "process_memory",
        }
    )
    # agent_workflow.add_edge("process_memory", "react_agent")
    # agent_workflow.add_edge("react_agent", "self_reflection")
    agent_workflow.add_edge("process_memory", "react_agent")
    agent_workflow.add_conditional_edges(
    "react_agent",
    is_self_reflection,
        {
            "finish_react_agent": "self_reflection",
            "not_finish_react_agent": END,
        }
    )
    agent_workflow.add_edge("self_reflection", END)

    plan_and_execute_app = agent_workflow.compile()
    
    try:
        png_data = plan_and_execute_app.get_graph().draw_mermaid_png()
        with open("workflow_diagram.png", "wb") as f:
            f.write(png_data)
        print("Graph saved as workflow_diagram.png")
    except Exception as e:
        print(f"PNG save failed: {e}")


    return plan_and_execute_app