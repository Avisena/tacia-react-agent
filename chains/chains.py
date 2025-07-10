from langchain_openai import ChatOpenAI 
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from typing import List
from tools.tools import ask_ai, interact_human, search_web
import streamlit as st
from langchain_openai import ChatOpenAI 
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import streamlit as st
from typing import List
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap, RunnableLambda
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain



openai_api_key = st.secrets["OPENAI_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

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

    process_memory_llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", api_key = openai_api_key)
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
    planner_llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", api_key = openai_api_key)

    planner = planner_prompt | planner_llm.with_structured_output(Plan)
    return planner

def create_react_agent_chain(callback_handler):

    prompt = hub.pull("hwchase17/react")
    prompt.template = """Kamu adalah konsultan pajak ahli yang ditugaskan untuk membantu klien berkonsultasi perihal perpajakan di Indonesia. Kamu akan memiliki percakapan/dialog dengan klien. Gunakan bahasa yang santai tapi profesional.
    
    Kamu punya akses ke tools berikut:
    {tools}

    Use the following format:

    Question: pertanyaan hukum yang perlu dijawab. Tujuan akhirmu adalah menjawab pertanyaan ini.   
    Thought: Berdasarkan Question di atas, pertimbangkan reasoning untuk menjawabnya dengan aturan perpajakan yang relevan dan informasi dari klien. 
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
    tools = [ask_ai, interact_human, search_web]
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", api_key = openai_api_key)
    agent = create_react_agent(llm,tools, prompt)
    agent_executor = AgentExecutor(agent=agent,tools=tools, handle_parsing_errors=True, verbose=True, return_intermediate_steps=True, callbacks=[callback_handler])
    return agent_executor

def create_self_reflection_chain():
    reflection_prompt = PromptTemplate.from_template("""
        Kamu telah memberikan jawaban berikut untuk pertanyaan terkait perpajakan:

        Pertanyaan: {question}
        Jawaban: {answer}

        Sekarang lakukan refleksi terhadap jawaban tersebut dengan mempertimbangkan:
        - Apakah dasar hukum atau peraturan pajak yang berlaku sudah dijelaskan?
        - Apakah logika penarikan kesimpulan sudah tepat dan sesuai konteks?
        - Apakah ada informasi penting atau pengecualian yang terlewat?
        - Apakah ada asumsi yang tidak dijelaskan secara eksplisit?

        Tuliskan refleksi singkat kamu sebagai konsultan pajak profesional.
        """)


    improvement_prompt = PromptTemplate.from_template("""
        Pertanyaan: {question}
        Jawaban Awal: {answer}
        Refleksi: {reflection}

        Sebagai konsultan pajak profesional, perbaiki jawaban awal di atas berdasarkan refleksi yang sudah dilakukan.
        Pastikan:
        - Menyebutkan aturan perpajakan yang relevan (misalnya, PP, UU, PMK).
        - Menjelaskan kewajiban atau pengecualian secara jelas.
        - Memberikan saran yang akurat dan mudah dipahami oleh klien.
        - Format penulisan markdown yang nyaman dibaca

        Tulis ulang jawaban direvisi dengan lebih baik. Tulis hanya jawaban revisinya tanpa yang lain.
        """)

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", api_key = openai_api_key)
    reflection_chain = RunnableMap({
        "question": lambda x: x["question"],
        "answer": lambda x: x["answer"]
    }) | reflection_prompt | llm | StrOutputParser()
    
    improvement_chain = RunnableMap({
        "question": lambda x: x["question"],
        "answer": lambda x: x["answer"],
        "reflection": reflection_chain
    }) | improvement_prompt | llm | StrOutputParser()

    self_reflection_chain = RunnableMap({
    "question": lambda x: x["question"],
    "answer": lambda x: x["answer"]
    }) | RunnableLambda(lambda x: {
        "question": x["question"],
        "reflection": reflection_chain.invoke(x),
        "improved_answer": improvement_chain.invoke(x)
    })

    return self_reflection_chain

def create_semantic_summary_chain():
    """
    Creates a LangChain LLMChain that generates a semantic summary 
    of a user message for long-term memory storage.
    
    Parameters:
        llm (BaseLanguageModel): Optional custom LLM. If None, uses ChatOpenAI.

    Returns:
        LLMChain: A LangChain chain that accepts 'user_input' and returns a summary.
    """

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", api_key = openai_api_key)
    prompt = """
        Kamu adalah agen AI yang bertugas merangkum informasi penting dari pengguna.

        Ringkas pesan berikut menjadi satu kalimat padat dan faktual yang menjelaskan kekhawatiran, tujuan, atau latar belakang pengguna. Hindari detail yang tidak penting atau pengulangan.

        Pesan pengguna:
        "{user_input}"

        Ringkasan:
        """ 
    semantic_summary_prompt = PromptTemplate(
        template=prompt,
        input_variables=["user_input"], 
    )
    semantic_summary = semantic_summary_prompt | llm
    return semantic_summary

process_memory_chain = create_memory_process_chain()
planner_chain = create_planner_chain()
self_reflection_chain = create_self_reflection_chain()
semantic_summary_chain = create_semantic_summary_chain()
# callback_handler = StopOnToolCallback(stop_on_tool="interact_human")
# react_agent_chain = create_react_agent_chain(callback_handler)