from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.embedder import Embedder
from src.search_engine import SearchEngine
import os

app = FastAPI(title="AI Intern Search Engine")

# Global instances
embedder = None
search_engine = None
DATA_DIR = "data/docs"

@app.on_event("startup")
def load_system():
    global embedder, search_engine
    
    # Ensure data exists
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print("WARNING: Data directory empty. Run download_data.py first.")
        return

    # 1. Preprocess and Load Embeddings [cite: 21]
    embedder = Embedder()
    print("Generating/Loading embeddings...")
    doc_data = embedder.generate_embeddings(DATA_DIR)
    
    # 2. Build Index [cite: 55]
    print("Building Vector Index...")
    search_engine = SearchEngine(doc_data)
    print("System Ready.")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search") # [cite: 74]
async def search_docs(request: SearchRequest):
    if not search_engine:
        raise HTTPException(status_code=503, detail="System initializing or data missing")
        
    # 1. Embed query [cite: 78]
    query_vec = embedder.embed_query(request.query)
    
    # 2. Search Index [cite: 79]
    results = search_engine.search(query_vec, request.top_k)
    
    # 3. Return results [cite: 80]
    return {"results": results}