from langchain_openai import ChatOpenAI 
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import tool
import streamlit as st


openai_api_key = st.secrets["OPENAI_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

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

        process_answer_llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", api_key = openai_api_key)
        process_answer_chain = process_answer_prompt | process_answer_llm | process_answer_parser
        answer = process_answer_chain.invoke({"question": question})
        return answer