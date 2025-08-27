import os
import hashlib
import tempfile
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "chroma_db"

def file_hash(file_path: str) -> str:
    """Generate a unique hash for file contents (duplicate detection)."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def save_uploaded_files(uploaded_files):
    """Save uploaded Streamlit files to temp directory and return their paths."""
    file_paths = []
    for file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                file_paths.append(tmp.name)
        except Exception as e:
            st.error(f"❌ Error saving file {file.name}: {e}")
    return file_paths

def load_vectorstore(uploaded_files):
    """Load or update Chroma vectorstore with PDF files, with error handling & deduplication."""
    paths = save_uploaded_files(uploaded_files)
    docs, seen_hashes = [], set()

    for path in paths:
        try:
            file_id = file_hash(path)
            if file_id in seen_hashes:
                st.warning(f"⚠️ Duplicate file skipped in this session: {os.path.basename(path)}")
                continue

            loader = PyPDFLoader(path)
            file_docs = loader.load()

            if not file_docs:
                st.warning(f"⚠️ No text extracted from {os.path.basename(path)}")
                continue

            docs.extend(file_docs)
            seen_hashes.add(file_id)

        except Exception as e:
            st.warning(f"⚠️ Skipping {os.path.basename(path)} due to error: {e}")

    if not docs:
        st.error("❌ No valid documents to process.")
        return None

    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_documents(docs)
    except Exception as e:
        st.error(f"❌ Error during text splitting: {e}")
        return None

    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    except Exception as e:
        st.error(f"❌ Failed to initialize embeddings: {e}")
        return None

    try:
        os.makedirs(PERSIST_DIR, exist_ok=True)

        # If DB already exists → update
        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
            vectorstore.add_documents(texts)
            st.success("✅ Vectorstore updated successfully!")
        else:
            # New DB
            vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=PERSIST_DIR
            )
            st.success("✅ New vectorstore created successfully!")

        return vectorstore

    except Exception as e:
        st.error(f"❌ Error creating/updating vectorstore: {e}")
        return None