import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import os
from src.embedder import Embedder
from src.search_engine import SearchEngine

# Page Config
st.set_page_config(page_title="AI Search Engine", layout="wide")
st.title("🔍 AI Document Search Engine")

# Sidebar
st.sidebar.header("Configuration")
use_expansion = st.sidebar.checkbox("Enable Query Expansion (Synonyms)", value=False)
top_k = st.sidebar.slider("Number of Results", 1, 20, 5)

# Initialize System (Cached to prevent reloading on every interaction)
@st.cache_resource
def load_system():
    data_dir = "data/docs"
    embedder = Embedder()
    
    # Load or Create Index
    if os.path.exists("vector_index.faiss"):
        search_engine = SearchEngine()
    else:
        st.info("Building Index... this may take a moment.")
        if not os.path.exists(data_dir):
             st.error("Data directory not found. Run download_data.py first.")
             return None, None
        doc_data = embedder.generate_embeddings(data_dir)
        search_engine = SearchEngine(doc_data)
        
    return embedder, search_engine

embedder, search_engine = load_system()

if not embedder or not search_engine:
    st.stop()

# Main Search Interface
query = st.text_input("Enter your search query:", placeholder="e.g., space exploration")

if query:
    with st.spinner("Searching..."):
        # Embed
        query_vec = embedder.embed_query(query, use_expansion=use_expansion)
        
        # Search
        results = search_engine.search(query_vec, top_k=top_k)
        
        # Display
        st.markdown(f"### Found {len(results)} results")
        
        for res in results:
            with st.expander(f"{res['doc_id']} (Score: {res['score']:.4f})"):
                st.markdown(f"**Preview:** {res['preview']}")
                st.markdown(f"**Why this result?** {res['explanation']['why_this']}")
                st.caption(f"Document Length: {res['explanation']['doc_length']} chars")