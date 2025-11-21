import sqlite3
import json
import hashlib
import os
from datetime import datetime

class CacheManager:
    def __init__(self, db_path="embeddings_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite table for caching[cite: 46, 48]."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embedding_cache (
                doc_id TEXT PRIMARY KEY,
                doc_hash TEXT,
                embedding TEXT,
                updated_at TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def get_embedding(self, doc_id, current_hash):
        """Retrieve embedding if hash matches, else return None[cite: 54]."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT doc_hash, embedding FROM embedding_cache WHERE doc_id=?", (doc_id,))
        result = cursor.fetchone()
        conn.close()

        if result:
            cached_hash, embedding_json = result
            if cached_hash == current_hash:
                return json.loads(embedding_json)
        return None

    def save_embedding(self, doc_id, current_hash, embedding):
        """Store embedding with timestamp[cite: 50, 51]."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        embedding_json = json.dumps(embedding)
        updated_at = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT OR REPLACE INTO embedding_cache (doc_id, doc_hash, embedding, updated_at)
            VALUES (?, ?, ?, ?)
        ''', (doc_id, current_hash, embedding_json, updated_at))
        conn.commit()
        conn.close()

    @staticmethod
    def compute_hash(text):
        """Generate SHA256 hash of text content[cite: 50]."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()