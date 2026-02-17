# scripts/init_memory.py
from pathlib import Path
import os
import chromadb
from sentence_transformers import SentenceTransformer
import logging

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MEMORY_DIR = PROJECT_ROOT / "memory" / "chroma_db"
EMBEDDER_DIR = PROJECT_ROOT / "memory" / "embedder_all-MiniLM-L6-v2"

embedder = None
collection = None

def init_local_memory(force=False):
    global embedder, collection
    os.makedirs(MEMORY_DIR, exist_ok=True)
    os.makedirs(EMBEDDER_DIR, exist_ok=True)
    
    embedder = SentenceTransformer(
        str(EMBEDDER_DIR),  # Convert Path to str
        device='mps'   # Apple Silicon – change to 'cuda' if GPU
    )
    
    client = chromadb.PersistentClient(path=str(MEMORY_DIR))
    collection = client.get_or_create_collection(
        name="atlas_memory",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Test insert if force or not exists
    if force or collection.count() == 0:
        text = "System initialization test entry for Atlas agent."
        embedding = embedder.encode(text).tolist()
        
        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=["test-init-001"],
            metadatas=[{"user_id": "system", "source": "init_memory.py"}]
        )
        print(f"Local Chroma initialized at: {MEMORY_DIR}")
        print("Test entry added.")
    
    # Retrieval test
    results = collection.query(
        query_texts=["Atlas agent initialization"],
        n_results=1,
        include=["documents", "metadatas", "distances"]
    )
    print("Retrieval test:", results)

# Run on import
init_local_memory()

if __name__ == "__main__":
    import os
    from sentence_transformers import SentenceTransformer
    
    EMBEDDER_DIR = PROJECT_ROOT / "memory" / "embedder_all-MiniLM-L6-v2"
    if not EMBEDDER_DIR.exists():
        print("Downloading embedder model...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        embedder.save(str(EMBEDDER_DIR))
        print("Embedder saved.")
    
    init_local_memory(force=True)