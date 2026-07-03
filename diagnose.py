"""
Full diagnostic: check ChromaDB state, data quality, and retrieval pipeline.
Run with: $env:PYTHONPATH="."; uv run python diagnose.py
"""
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

import json
import numpy as np
from src.vectorstore import get_embed_model, get_chroma_client, get_collection

def main():
    print("=" * 60)
    print("HIREVIEW DIAGNOSTIC")
    print("=" * 60)

    # 1. Check ChromaDB
    client = get_chroma_client()
    collection = get_collection(client)
    count = collection.count()
    print(f"\n[1] ChromaDB count: {count}")

    if count == 0:
        print("ERROR: ChromaDB is empty! Need to ingest.")
        return

    # 2. Peek at stored documents
    sample = collection.peek(limit=2)
    print(f"\n[2] Sample document snippet (first 300 chars):")
    for i, doc in enumerate(sample["documents"]):
        print(f"  Doc {i}: {repr(doc[:300])}")
    print(f"\n  Sample metadata keys: {list(sample['metadatas'][0].keys()) if sample['metadatas'] else 'None'}")
    print(f"  Sample metadata: {sample['metadatas'][0] if sample['metadatas'] else 'None'}")

    # 3. Check what Categories are stored
    print("\n[3] Checking category distribution...")
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    cats = {}
    sources = {}
    for m in all_meta:
        c = m.get("Category", "MISSING")
        s = m.get("Source", "MISSING")
        cats[c] = cats.get(c, 0) + 1
        sources[s] = sources.get(s, 0) + 1
    print("  Categories (top 10):", dict(sorted(cats.items(), key=lambda x: -x[1])[:10]))
    print("  Sources:", sources)

    # 4. Test retrieval with a real JD
    print("\n[4] Testing retrieval...")
    model = get_embed_model()
    jd = "Data Scientist with Python, pandas, scikit-learn, machine learning, statistics, SQL"
    emb = model.encode([jd])[0].tolist()

    results = collection.query(query_embeddings=[emb], n_results=10,
                               include=["metadatas", "documents", "distances"])
    
    print(f"  Query: '{jd}'")
    print(f"  Top 10 results:")
    for i in range(len(results["distances"][0])):
        dist = results["distances"][0][i]
        sim = 1 - dist
        meta = results["metadatas"][0][i]
        doc_snippet = results["documents"][0][i][:100]
        print(f"    #{i+1}: {meta.get('Name')} | Cat: {meta.get('Category')} | dist={dist:.4f} | sim={sim:.2%} | doc: {repr(doc_snippet)}")

    print("\n[5] Checking if embedding model works correctly...")
    test1 = "Python data scientist machine learning"
    test2 = "Python data scientist machine learning"
    test3 = "Cook chef restaurant kitchen food"
    e1, e2, e3 = model.encode([test1, test2, test3])
    sim_same = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    sim_diff = np.dot(e1, e3) / (np.linalg.norm(e1) * np.linalg.norm(e3))
    print(f"  Same text similarity: {sim_same:.4f} (should be ~1.0)")
    print(f"  Different text similarity: {sim_diff:.4f} (should be low)")

    print("\n[DONE]")

if __name__ == "__main__":
    main()
