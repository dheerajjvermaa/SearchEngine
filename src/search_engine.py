import faiss
import numpy as np
import os
import pickle

class SearchEngine:
    def __init__(self, doc_data=None, index_path="vector_index.faiss", metadata_path="metadata.pkl"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.doc_map = {}
        
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path) and doc_data is None:
            self.load_index()
        elif doc_data:
            self.doc_data = doc_data
            self.doc_map = {i: doc for i, doc in enumerate(doc_data)}
            self._build_index()
            self.save_index()

    def _build_index(self):
        if not self.doc_data:
            return
        
        # Ensure float32 for FAISS
        embeddings = np.array([d['embedding'] for d in self.doc_data]).astype('float32')
        faiss.normalize_L2(embeddings)
        
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.doc_map, f)

    def load_index(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "rb") as f:
            self.doc_map = pickle.load(f)

    def search(self, query_embedding, top_k=5):
        # Ensure input is 2D float32 array
        query_vec = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vec)
        
        scores, indices = self.index.search(query_vec, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            doc = self.doc_map[idx]
            
            results.append({
                "doc_id": doc['doc_id'],
                "score": float(score),
                "preview": doc['text'][:200] + "...",
                "explanation": {
                    "why_this": f"Similarity score {score:.4f}",
                    "doc_length": doc['length']
                }
            })
        return results