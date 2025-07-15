import streamlit as st
from agents.taxation_agent import create_agent
from langchain.callbacks import get_openai_callback
import markdown
import json
import io

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
        <h1 class="title">💬 Konsultan Hukum Pidana AI</h1>
        <p class="subtitle">Tanyakan apa pun tentang kasus anda</p>
    </div>
    <style>
        .hero-container {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 140px;
            background-color: white;
            z-index: 1000;
            border-bottom: 1px solid #ddd;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1.title {
            text-align: center;
            margin-bottom: 0.2em;
            font-size: 2rem;
            margin-top: 0;
        }
        p.subtitle {
            text-align: center;
            color: gray;
            margin-top: 0;
            font-size: 1rem;
            margin-bottom: 0;
        }

        .cost-display-header {
            position: fixed;
            top: 85px;
            right: 20px;
            font-size: 0.9rem;
            color: #888;
            z-index: 1001;
            background-color: white;
            padding: 0.2rem 0.5rem;
        }

        .main-content {
            margin-top: 160px; /* Space for fixed header */
            padding-bottom: 100px; /* Space for input at bottom */
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
        .assistant-reasoning {
            background-color: #FFF8DC; /* light yellow */
            font-style: italic;
            border-left: 4px dashed #D4A017;
            padding-left: 1rem;
            margin-right: auto;
            font-size: 0.95rem;
            color: #555;
        }

        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: white;
            padding: 1rem;
            border-top: 1px solid #ddd;
            z-index: 999;
        }

        .cost-display {
            text-align: right;
            font-size: 0.9rem;
            color: #888;
            margin-bottom: 1rem;
            display: none; /* Hide this since it's now in header */
        }
    </style>
""", unsafe_allow_html=True)

# --- COST DISPLAY IN HEADER ---
st.markdown(f"""
    <div class="cost-display-header">
        💵 Total Biaya Penggunaan: <strong>Rp. {st.session_state.total_cost * 16200:,.0f}</strong>
    </div>
""", unsafe_allow_html=True)

# # --- MAIN CONTENT AREA ---
# st.markdown('<div class="main-content">', unsafe_allow_html=True)

# --- COST DISPLAY ---
st.markdown(f"""
    <div class="cost-display">
        💵 Total Biaya Penggunaan: <strong>Rp. {st.session_state.total_cost * 16200:,.0f}</strong>
    </div>
""", unsafe_allow_html=True)

# --- CHAT DISPLAY ---
chat_container = st.container()
with chat_container:    
    # Get the current chat history
    current_chat_history = st.session_state.last_result.get("chat_history", [])
    
    # Display each message
    for i, msg in enumerate(current_chat_history):
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        # Skip empty messages
        if not content:
            continue
            
        is_user = role == "user"

        if role == "assistant_reasoning":
            bubble_class = "assistant-reasoning"
            prefix = "🧠 I think: "
        else:
            bubble_class = "user" if is_user else "assistant"
            prefix = ""

        # Create columns for alignment
        cols = st.columns([0.3, 0.7]) if is_user else st.columns([0.7, 0.3])

        with cols[1] if is_user else cols[0]:
            # Safely render markdown content
            try:
                rendered_content = f"<pre>{content}</pre>"
            except Exception as e:
                rendered_content = content  # Fallback to plain text
                
            st.markdown(
                f"<div class='chat-bubble {bubble_class}'>{prefix}{rendered_content}</div>",
                unsafe_allow_html=True
            )
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close chat-wrapper

st.markdown('</div>', unsafe_allow_html=True)  # Close main-content

# --- FIXED INPUT CONTAINER ---
st.markdown('<div class="input-container">', unsafe_allow_html=True)

question = st.chat_input("Tulis pertanyaan hukum pidana Anda...")

st.markdown('</div>', unsafe_allow_html=True)  # Close input-container

# --- PROCESS NEW QUESTION ---
if question:
    # Get previous chat history
    previous_chat = st.session_state.last_result.get("chat_history", [])
    
    # Add user message to chat history
    current_chat = previous_chat + [{"role": "user", "content": question}]
    
    # Show processing indicator
    with st.spinner("🤔 Tacia sedang berpikir..."):
        try:
            with get_openai_callback() as cb:
                result = st.session_state.agent.invoke({
                    "chat_history": current_chat,
                    "intermediate_steps": st.session_state.last_result.get("intermediate_steps", []),
                    "curr_state": st.session_state.last_result.get("curr_state", []),
                    "memory_based_question": st.session_state.last_result.get("memory_based_question", "")
                })
                
                # Debug: Print result structure
                print("RESULT KEYS:", result.keys() if isinstance(result, dict) else "Not a dict")
                print("RESULT:", result)
                
                # Update session state
                st.session_state.last_result = result
                st.session_state.total_cost += cb.total_cost
                
                # Force rerun to update display
                st.rerun()
                
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            print(f"Error: {e}")

# --- DEBUG INFO (uncomment for debugging) ---
# with st.expander("Debug Info"):
#     st.write("Session State Keys:", list(st.session_state.keys()))
#     st.write("Last Result:", st.session_state.last_result)
#     st.write("Chat History Length:", len(st.session_state.last_result.get("chat_history", [])))
#     if st.session_state.last_result.get("chat_history"):
#         st.write("Last Message:", st.session_state.last_result["chat_history"][-1])