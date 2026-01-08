from langchain_openai import ChatOpenAI 
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
import streamlit as st


openai_api_key = st.secrets["OPENAI_API_KEY"]
# groq_api_key = st.secrets["GROQ_API_KEY"]

@tool
def interact_with_human(message):
    """
    1.  Gunakan tool ini ketika kamu membutuhkan informasi tambahan dari manusia yang tidak tersedia dalam pertanyaan awal. Misalnya, saat informasi tidak lengkap, ambigu, atau butuh klarifikasi lanjutan dari penanya untuk bisa menjawab dengan akurat. Tool ini akan mengajukan pertanyaan langsung ke manusia.

        Tool ini sangat penting untuk menghindari asumsi yang salah dalam konteks perpajakan.

        Contoh penggunaan:
        - Menanyakan sumber penghasilan jika hanya disebut "penghasilan 50 juta".
        - Meminta rincian apakah transaksi termasuk PPN.
        - Klarifikasi apakah donasi dilakukan ke lembaga resmi yang diakui negara.

    2. Gunakan tool ini untuk memberikan penjelasan terkait kepada pengguna.

    Args:
        message (str): The message to deliver to the human.

    Returns:
        str: The human's input.
    """
    return input(f"{message} ")

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

@tool
def search_web(query):
    """
    Pengetahuan anda hanya sampai tahun 2023. Sekarang adalah Juli 2025.
    Gunakan ini untuk mencari informasi up-to-date dari internet terkait aturan terbaru, pasal terbaru, dll.
    query(str): the query
    """
    tool = DuckDuckGoSearchResults()
    results = tool.run(query)
    print(f"Web result: {results}")
    return(results)
