# --- FIX for Streamlit Deployment ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- END OF FIX ---

import streamlit as st
from ai_core_v2 import MuskTwinV2
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Elon Musk Digital Twin",
    page_icon="🚀",
    layout="wide"
)

# --- Title and Description ---
st.title("AI Digital Twin: Elon Musk 🚀")
st.markdown("Ask a question and get an answer in the voice and style of Elon Musk, based on his public statements, interviews, and writings.")

# --- Caching the AI Model ---
@st.cache_resource
def load_ai_twin():
    """Loads the MuskTwinV2 instance and caches it for the session."""
    print("Loading AI core for the first time...")
    return MuskTwinV2()

# --- Main Application Logic ---
try:
    twin = load_ai_twin()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Using the standard st.audio component which does not autoplay
            if "audio" in message and message["audio"]:
                st.audio(io.BytesIO(message["audio"]), format="audio/mp3")
            if "sources" in message and message["sources"]:
                st.info(f"Sources: {', '.join(message['sources'])}")

    user_question = st.chat_input("Ask Elon a question...")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Elon is thinking..."):
                response = twin.ask(user_question)
                answer = response['answer']
                sources = response['sources']
                audio = response['audio']
                
                st.markdown(answer)
                
                # Using the standard st.audio component which does not autoplay
                if audio:
                    st.audio(io.BytesIO(audio), format="audio/mp3")
                
                if sources:
                    st.info(f"Sources: {', '.join(sources)}")
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "sources": sources,
            "audio": audio
        })

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error("The AI core might be missing. Please ensure you have run `python ai_core_v2.py` first to build the database.")
