import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path, override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print(f"GROQ_API_KEY loaded: {GROQ_API_KEY}")

if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("⚠️ GROQ_API_KEY not found. Please set it in .env")

def get_llm_chain(vectorstore):
    llm = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="openai/gpt-oss-120b",
        temperature=0.3,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )