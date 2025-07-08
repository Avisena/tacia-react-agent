from langchain_openai import ChatOpenAI 
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.agents.agent import AgentAction, AgentFinish
from langgraph.graph import END, StateGraph
from pprint import pprint
import streamlit as st
from typing_extensions import TypedDict
from typing import Any, List, Tuple, TypedDict
from langchain.tools import tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent


### Helper functions for the notebook
from langchain.callbacks.base import BaseCallbackHandler
from nodes.nodes import planner, process_memory, react_agent
from helpers.helpers import get_last_tool_input, insert_observation_for_last_interact_human, clean_agent_log, get_last_user_message

openai_api_key = st.secrets["OPENAI_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

class StopOnToolCallback(BaseCallbackHandler):
    def __init__(self, stop_on_tool: str):
        self.stop_on_tool = stop_on_tool
        self.intermediate_steps: List[Tuple[AgentAction, str]] = []
        self.pending_action: AgentAction | None = None

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        print(f"[Callback] Agent chose tool: {action.tool}")
        self._latest_action = action

        self.intermediate_steps.append((self._latest_action))

        # Stop only on the specified tool
        if self._latest_action.tool == self.stop_on_tool:
            print(f"[Stop] Tool '{self.stop_on_tool}' called with input: {self._latest_action.tool_input}")
            raise KeyboardInterrupt("Stopped after specified tool was used.")
    
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

class Plan(BaseModel):
        """Plan to follow in future"""

        steps: List[str] = Field(
            description="langkah-langkah terurut yang harus diambil untuk melakukan permintaan pelanggan"
        )

def create_memory_process_chain():
    class MemoryProcess(BaseModel):
        """Remake question based on memory."""
        memory_based_question : str = Field(description="processed memory.")

    process_memory_parser = JsonOutputParser(pydantic_object=MemoryProcess)


    process_memory_prompt_template = """ 
    Riwayat percakapan:
    {chat_history}

    Tugas Anda adalah memeriksa dari riwayat percakapan apakah pertanyaan saat ini adalah lanjutan dari riwayat percakapan.
    - Jika **YA**, reformulasikan pertanyaan agar berdiri sendiri.
    - Jika **TIDAK**, kembalikan pertanyaan apa adanya tanpa perubahan.

    Berikan hanya satu pertanyaan akhir sebagai hasil.
    {format_instructions}
   """

    process_memory_prompt = PromptTemplate(
        template=process_memory_prompt_template,
        input_variables=["question", "chat_history"],
        partial_variables={"format_instructions": process_memory_parser.get_format_instructions()},
    )

    process_memory_llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=2000, api_key = openai_api_key)
    process_memory_chain = process_memory_prompt | process_memory_llm | process_memory_parser
    return process_memory_chain

def create_planner_chain():
    planner_prompt =""" 
    Anda adalah seorang konsultan pajak di Indonesia.
    Anda menerima pertanyaan dari klien: {memory_based_question}.
    Buatlah planning penalaran dengan cara berpikir seorang konsultan pajak yang tersusun atas apa yang harus anda cari tahu atau lakukan untuk menjawab pertanyaan klien.
    Jangan menambahkan langkah yang tidak perlu.
    Hasil dari langkah terakhir harus berupa jawaban akhir.
    Pastikan setiap langkah memiliki semua informasi yang dibutuhkan — jangan melewatkan langkah apa pun.
    """

    planner_prompt = PromptTemplate(
        template=planner_prompt,
        input_variables=["memory_based_question"], 
        )
    planner_llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=2000, api_key = openai_api_key)

    planner = planner_prompt | planner_llm.with_structured_output(Plan)
    return planner

def create_react_agent_chain(callback_handler):
    @tool
    def ask_ai(question):
        """
        Untuk meminta AI informasi yang Anda butuhkan, fungsi ini menerima pertanyaan sebagai argumen.
        question(str): the question
        """
        class AnswerProcess(BaseModel):
            """Remake question based on memory."""
            answer : str = Field(description="the answer.")

        process_answer_parser = JsonOutputParser(pydantic_object=AnswerProcess)
        process_answer_template = """ 
        Kamu adalah konsultan pajak di Indonesia. Jawablah permintaan di bawah ini:
        {question}

        Jawablah dengan dasar hukum yang jelas disertai pasalnya.
        Berikan hanya satu jawaban akhir sebagai hasil.
        {format_instructions}
        """

        process_answer_prompt = PromptTemplate(
            template=process_answer_template,
            input_variables=["question", "chat_history"],
            partial_variables={"format_instructions": process_answer_parser.get_format_instructions()},
        )

        process_answer_llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=2000, api_key = openai_api_key)
        process_answer_chain = process_answer_prompt | process_answer_llm | process_answer_parser
        answer = process_answer_chain.invoke({"question": question})
        return answer

    @tool
    def interact_human(question):
        """
        Gunakan ini hanya jika Anda perlu berinteraksi dengan manusia.
        1. Contoh ketika input dari pengguna tidak memuat cukup informasi pribadi mereka dan Anda perlu menanyakan lebih lanjut.
        Contohnya: gaji mereka, preferensi mereka, kota tempat tinggal mereka, dan sebagainya.

        2. Gunakan ini untuk memberikan penjelasan dan saran kepada pengguna.
        Args:
            question (str): The question to ask the human.

        Returns:
            str: The human's input.
        """
        return input(f"{question} ")

    prompt = hub.pull("hwchase17/react")
    prompt.template = """Kamu adalah konsultan pajak ahli yang ditugaskan untuk membantu klien berkonsultasi perihal perpajakan di Indonesia. Gunakan bahasa yang santai tapi profesional.
    
    Kamu punya akses ke tools berikut:
    {tools}

    Use the following format:

    Question: pertanyaan hukum yang perlu dijawab. Tujuan akhirmu adalah menjawab pertanyaan ini.   
    Thought: Pahami masalah dan situasi klien. Pikirkan langkah demi langkah terstruktur tentang isu hukum yang dimaksud menggunakan pendekatan cara berpikir konsultan hukum. Berpikir lah secara luas. Selalu gunakan dasar hukum sebagai acuan.
    Action: tindakan yang akan dilakukan berdasarkan langkah demi langkah yang telah kamu buat. Harus salah satu dari [{tool_names}]  
    Action Input: Input untuk Action
    Observation: Umpan balik dari action input
    ... (this Thought/Action/Action Input/Observation can repeat N times)  
    Thought: I now know the answer.
    Final Answer: Jawaban akan pertanyaan. Tuangkan hasil reasoning anda dari Thought disertai pasal hukumnya secara terstruktur dan rapi.

    Begin!

    Question: {input}  
    Thought: {agent_scratchpad}
    """
    tools = [ask_ai, interact_human]
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=2000, api_key = openai_api_key)
    agent = create_react_agent(llm,tools, prompt)
    agent_executor = AgentExecutor(agent=agent,tools=tools, handle_parsing_errors=True, verbose=True, return_intermediate_steps=True, callbacks=[callback_handler])
    return agent_executor



process_memory_chain = create_memory_process_chain()
planner_chain = create_planner_chain()


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
    if(state['curr_state'] != 'processing_react_agent'):
        state["curr_state"] = "process_memory"
        process_memory_output = process_memory_chain.invoke(input_values)
        memory_based_question = process_memory_output["memory_based_question"]
        print(f'memory_based_question: {memory_based_question}')
        state['memory_based_question'] = memory_based_question

    if(state['curr_state'] == 'processing_react_agent'):
        print("INTERMEDIATE_STEPS: ", state['intermediate_steps'])
        print("CHAT_HISTORY: ", state['chat_history'])
        print(get_last_user_message(state['chat_history']))
        state['intermediate_steps'] = insert_observation_for_last_interact_human(state['intermediate_steps'], get_last_user_message(state['chat_history']))
        print("INSERT ANSWER TO INT STEPS: ", state['intermediate_steps'])
    return state

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
            state["chat_history"].append({"role": "assistant", "content": clean_agent_log(answer["intermediate_steps"][-1][0].log)})
        print("ANSWER: ", answer)
        state['response'] = answer['output']
        state["chat_history"].append({"role": "assistant", "content": state["response"]})
        state["curr_state"] = "finish_react_agent"
        state["intermediate_steps"] = []
    except KeyboardInterrupt:
        print("Agent stopped after hitting tool:", callback_handler.stop_on_tool)
        print(callback_handler.intermediate_steps)
        state['intermediate_steps'].extend(callback_handler.intermediate_steps)
        print(f"Total saved steps length: {len(state['intermediate_steps'])}")
        print(f"Total saved steps: {state['intermediate_steps']}")
        state['chat_history'].append({"role": "assistant", "content": get_last_tool_input(callback_handler.intermediate_steps)})
    return state

def create_agent():
    agent_workflow = StateGraph(PlanExecute)

    # Add the anonymize node
    agent_workflow.add_node("process_memory", process_memory)
    agent_workflow.add_node("planner", planner)
    agent_workflow.add_node("react_agent", react_agent)
    # Set the entry point
    agent_workflow.set_entry_point("process_memory")
    agent_workflow.add_edge("process_memory", "react_agent")
    agent_workflow.add_edge("react_agent", END)

    plan_and_execute_app = agent_workflow.compile()

    return plan_and_execute_app