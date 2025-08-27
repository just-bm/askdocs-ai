import streamlit as st
from langchain_community.vectorstores import Chroma

def inspect_chroma(vectorstore):
    st.sidebar.markdown("🧪 **ChromaDB Inspector**")
    k = st.sidebar.number_input("📄 Number of top results to fetch", min_value=1, max_value=10, value=3, step=1)

    try:
        doc_count = vectorstore._collection.count()
        st.sidebar.success(f"🔎 {doc_count} documents stored in ChromaDB.")
    except Exception as e:
        st.sidebar.error("Could not fetch document count.")
        st.sidebar.code(str(e))

    query = st.sidebar.text_input("🔍 Test a query against ChromaDB")

    if query:
        try:
            results = vectorstore.similarity_search(query, k=k)
            st.sidebar.markdown("### Top Matching Chunks:")
            for i, doc in enumerate(results):
                st.sidebar.markdown(f"**Result {i+1}:**")
                st.sidebar.markdown(doc.page_content[:300] + "...")
                st.sidebar.markdown("---")
        except Exception as e:
            st.sidebar.error("Error querying ChromaDB")
            st.sidebar.code(str(e))