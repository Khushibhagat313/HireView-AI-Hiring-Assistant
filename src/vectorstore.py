import os
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from src.config import EMBED_MODEL, CHROMA_DIR, COLLECTION

# Global vars for caching the model
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model

def get_chroma_client():
    if not os.path.exists(CHROMA_DIR):
        os.makedirs(CHROMA_DIR)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client

def get_collection(client):
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def add_documents(documents, metadatas, ids):
    model = get_embed_model()
    embeddings = model.encode(documents, show_progress_bar=False, batch_size=32).tolist()
    
    client = get_chroma_client()
    collection = get_collection(client)
    
    collection.upsert(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

def query_resumes(jd_text, n_results=50, threshold=0.0, where=None):
    model = get_embed_model()
    # BGE models need "Represent this sentence: " prefix on the QUERY side for best retrieval performance
    query_text = f"Represent this sentence: {jd_text}"
    query_embedding = model.encode([query_text]).tolist()
    
    client = get_chroma_client()
    collection = get_collection(client)
    
    # Make sure we don't ask for more than what's stored
    total = collection.count()
    n_results = min(n_results, total)
    
    kwargs = dict(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["metadatas", "documents", "distances", "embeddings"]
    )
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)
    
    candidates = []
    if results["distances"] and len(results["distances"]) > 0:
        for i, distance in enumerate(results["distances"][0]):
            # Cosine distance → similarity = 1 - distance
            similarity = 1.0 - distance
            if similarity >= threshold:
                candidates.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": similarity,
                    "score_pct": int(round(similarity * 100)),
                    "embedding": results["embeddings"][0][i]
                })
    
    # Sort descending by score
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return candidates
