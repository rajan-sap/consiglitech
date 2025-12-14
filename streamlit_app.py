import streamlit as st
from generation.generator import generate_answer

# Set up the Streamlit page and center the chatbot title
st.set_page_config(page_title="Hybrid RAG Chatbot", layout="centered")
st.markdown("""
<h1 style='text-align: center; margin-bottom: 1.5em;'>Hybrid RAG Chatbot</h1>
""", unsafe_allow_html=True)

# Initialize session state variables for chat history, input value, and response status
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'input_value' not in st.session_state:
    st.session_state.input_value = ""
if 'awaiting_response' not in st.session_state:
    st.session_state.awaiting_response = False

# Function to handle sending a user message
def send_message():
    user_input = st.session_state.input_value
    if user_input and user_input.strip():
        # Only append if this is a new message (avoid double appending on rerun)
        if not (len(st.session_state.chat_history) > 0 and st.session_state.chat_history[-1]["role"] == "user" and st.session_state.chat_history[-1]["content"] == user_input):
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.input_value = ""  # Clear input
            st.session_state.awaiting_response = True

# Display the chat history (user and AI messages)
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div style='text-align:right; margin:8px 0;'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align:left; margin:8px 0;'><b>AI:</b> {msg['content']}</div>", unsafe_allow_html=True)

# If a response is pending, show a spinner, generate the answer, and display it
pending_answer = None
if st.session_state.awaiting_response:
    with st.spinner("AI is looking for information..."):
        user_msg = st.session_state.chat_history[-1]["content"]
        pending_answer = generate_answer(user_msg)
        # Show the answer immediately below the last user message
        st.markdown(f"<div style='text-align:left; margin:8px 0;'><b>AI:</b> {pending_answer}</div>", unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "assistant", "content": pending_answer})
        st.session_state.awaiting_response = False

# Add a gap between chat and input area, then show the input area at the bottom
st.markdown("<div style='height: 4em;'></div>", unsafe_allow_html=True)
cols = st.columns([8,1])
with cols[0]:
    # Text input for user queries (Enter key triggers send_message)
    st.text_input(
        "Your question:",
        value=st.session_state.input_value,
        key="input_value",
        placeholder="Type your question...",
        label_visibility="collapsed",
        on_change=send_message
    )
with cols[1]:
    # Send button (click triggers send_message)
    st.button("▶", key="send_btn", help="Send", use_container_width=True, on_click=send_message)