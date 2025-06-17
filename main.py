import warnings
import logging
import streamlit as st
from modules.chat import display_chat_history, handle_user_input, download_chat_history
from modules.pdf_handler import upload_pdfs
from modules.vectorstore import load_vectorstore
from modules.llm import get_llm_chain
from modules.chroma_inspector import inspect_chroma

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(
    page_title="RagBot: Your AI Knowledge Assistant",
    page_icon=""
)

st.title("AskDocs AI: AI-Powered PDF Q&A Bot")
st.caption("Upload PDFs, ask questions, and get instant answers with context-aware responses.")

with st.sidebar:
    st.markdown("""
    **AskDocs AI** is an AI-powered chatbot that leverages **RAG (Retrieval-Augmented Generation)** to answer your questions based on the content of uploaded PDFs.
    🔗 [GitHub](https://github.com/Balaji-R-05) 
    """)

uploaded_files, submitted = upload_pdfs()

if submitted and uploaded_files:
    with st.spinner("🔄 Updating vector database..."):
        vectorstore = load_vectorstore(uploaded_files)
        st.session_state.vectorstore = vectorstore

if "vectorstore" in st.session_state:
    inspect_chroma(st.session_state.vectorstore)

display_chat_history()

if "vectorstore" in st.session_state:
    handle_user_input(get_llm_chain(st.session_state.vectorstore))

download_chat_history()