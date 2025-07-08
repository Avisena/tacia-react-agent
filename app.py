import streamlit as st
from agents.taxation_agent import create_agent
from langchain.callbacks import get_openai_callback
import markdown

# --- AGENT SETUP ---
if "agent" not in st.session_state:
    st.session_state.agent = create_agent()

if "last_result" not in st.session_state:
    st.session_state.last_result = {
        "chat_history": [],
        "intermediate_steps": [],
        "curr_state": [],
        "memory_based_question": ""
    }

if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

# --- PAGE HEADER ---
st.markdown("""
    <div class="hero-container">
        <h1 class="title">💬 Konsultan Pajak AI</h1>
        <p class="subtitle">Tanyakan apa pun tentang PPh, PPN, atau kewajiban pajak Anda.</p>
    </div>
    <style>
        .hero-container {
            position: relative;
            height: 200px;
        }
        h1.title {
            text-align: center;
            margin-bottom: 0.2em;
            font-size: 2rem;
        }
        p.subtitle {
            text-align: center;
            color: gray;
            margin-top: 0;
            font-size: 1rem;
        }

        .chat-wrapper {
            height: 500px;
            overflow-y: auto;
            padding: 1rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #ffffff;
            margin-bottom: 1rem;
        }

        .chat-bubble {
            border-radius: 1rem;
            padding: 0.75rem 1rem;
            font-size: 1.05rem;
            line-height: 1.5;
            display: inline-block;
            max-width: 100%;
            margin: 20px;
            word-wrap: break-word;
        }

        .user {
            background-color: #DCF8C6;
            text-align: right;
            margin-left: auto;
            width: fit-content;
            float: right;
        }

        .assistant {
            background-color: #F1F0F0;
            text-align: left;
            margin-right: auto;
        }
    </style>
""", unsafe_allow_html=True)


# --- CHAT INPUT ---
question = st.chat_input("Tulis pertanyaan pajak Anda...")

if question:
    previous_chat = st.session_state.last_result.get("chat_history", [])
    current_chat = previous_chat + [{"role": "user", "content": question}]

    with get_openai_callback() as cb:
        result = st.session_state.agent.invoke({
            "chat_history": current_chat,
            "intermediate_steps": st.session_state.last_result.get("intermediate_steps", []),
            "curr_state": st.session_state.last_result.get("curr_state", []),
            "memory_based_question": st.session_state.last_result.get("memory_based_question", "")
        })
        print("RESULT: ", result)
        st.session_state.last_result = result
        st.session_state.total_cost += cb.total_cost

# --- COST DISPLAY UNDER HERO ---
st.markdown(f"""
    <p style='text-align:right; font-size:0.9rem; color:#888; margin-top:-1rem;'>
        💵 Total Biaya Penggunaan: <strong>Rp. {st.session_state.total_cost * 16200:,.0f}</strong>
    </p>
""", unsafe_allow_html=True)

# --- RENDER CHAT BUBBLES ---
for msg in st.session_state.last_result.get("chat_history", []):
    is_user = msg["role"] == "user"
    bubble_class = "user" if is_user else "assistant"
    cols = st.columns([0.3, 0.7]) if is_user else st.columns([0.7, 0.3])

    with cols[1] if is_user else cols[0]:
        st.markdown(
            f"<div class='chat-bubble {bubble_class}'>{markdown.markdown(msg['content'])}</div>",
            unsafe_allow_html=True
        )
