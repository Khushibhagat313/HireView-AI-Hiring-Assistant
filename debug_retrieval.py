import sys
from src.vectorstore import get_embed_model, get_chroma_client, get_collection
import pandas as pd

def debug():
    jd = "Data Science Fresher with Python, SQL, Machine Learning, Pandas, Numpy, Scikit-learn, Matplotlib."
    client = get_chroma_client()
    collection = get_collection(client)
    
    print(f"Total docs in Chroma: {collection.count()}")
    
    model = get_embed_model()
    emb = model.encode(jd).tolist()
    
    results = collection.query(query_embeddings=[emb], n_results=5)
    
    distances = results["distances"][0] if results["distances"] else []
    print("Top 5 distances:", distances)
    
    for i, dist in enumerate(distances):
        sim = 1 - dist
        meta = results["metadatas"][0][i]
        print(f"Candidate {i+1}: {meta.get('Name')} | dist: {dist:.4f} | sim: {sim:.4f}")

if __name__ == "__main__":
    debug()
