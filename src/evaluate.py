from src.embedder import Embedder
from src.search_engine import SearchEngine
import time

def evaluate_system():
    print("--- Starting Evaluation ---")
    
    # Initialize
    embedder = Embedder()
    # Assumes index already built by previous run
    search_engine = SearchEngine() 
    
    test_queries = [
        "space mission",
        "computer graphics algorithms",
        "political middle east conflict",
        "windows operating system"
    ]
    
    total_time = 0
    
    print(f"\nRunning {len(test_queries)} test queries...\n")
    
    for q in test_queries:
        start = time.time()
        q_vec = embedder.embed_query(q)
        results = search_engine.search(q_vec, top_k=3)
        duration = time.time() - start
        total_time += duration
        
        print(f"Query: '{q}' ({duration:.4f}s)")
        for r in results:
            print(f"  - [{r['score']:.4f}] {r['doc_id']}")
            
    avg_time = total_time / len(test_queries)
    print(f"\nAverage Latency: {avg_time:.4f}s")
    print("--- Evaluation Complete ---")

if __name__ == "__main__":
    evaluate_system()