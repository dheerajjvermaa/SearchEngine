import os
import re
import numpy as np
import nltk
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
from src.cache_manager import CacheManager

# Download WordNet silently
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

class Embedder:
    def __init__(self, cache_db_path="embeddings_cache.db"):
        print("Loading Recommended Model (all-MiniLM-L6-v2)...")
        # Using the specific model recommended in Assignment 
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache_manager = CacheManager(cache_db_path)

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def expand_query(self, query):
        """Query expansion using WordNet."""
        words = query.split()
        expanded_words = set(words)
        for word in words:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if "_" not in lemma.name():
                        expanded_words.add(lemma.name())
        return " ".join(list(expanded_words))

    def generate_embeddings(self, data_dir, batch_size=32):
        doc_data = []
        files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        
        texts_to_embed = []
        metadata_list = []
        
        print(f"Processing {len(files)} files...")
        
        for filename in files:
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            clean_text = self.preprocess_text(raw_text)
            doc_hash = self.cache_manager.compute_hash(clean_text)
            doc_id = filename.split('.')[0]
            
            # Check cache
            cached_embedding = self.cache_manager.get_embedding(doc_id, doc_hash)
            
            if cached_embedding:
                doc_data.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "text": clean_text,
                    "embedding": cached_embedding,
                    "length": len(clean_text)
                })
            else:
                texts_to_embed.append(clean_text)
                metadata_list.append((doc_id, doc_hash, filename, clean_text))

        # Batch Process New Texts
        if texts_to_embed:
            print(f"Encoding {len(texts_to_embed)} new documents...")
            # SentenceTransformers handles batching internally
            embeddings = self.model.encode(texts_to_embed, batch_size=batch_size, show_progress_bar=True)
            
            # Convert numpy array to list for storage
            embeddings = embeddings.tolist()
            
            for j, embedding in enumerate(embeddings):
                doc_id, doc_hash, filename, text = metadata_list[j]
                self.cache_manager.save_embedding(doc_id, doc_hash, embedding)
                
                doc_data.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "text": text,
                    "embedding": embedding,
                    "length": len(text)
                })
                    
        return doc_data

    def embed_query(self, query, use_expansion=False):
        if use_expansion:
            query = self.expand_query(query)
            print(f"Expanded Query: {query}")
            
        clean_query = self.preprocess_text(query)
        # Encode returns numpy array, no extra tensor conversion needed
        return self.model.encode([clean_query])[0]