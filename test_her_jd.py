import sys
from src.vectorstore import get_embed_model, get_chroma_client, get_collection
import pandas as pd

def debug():
    jd = """We are hiring a motivated and curious fresher to join our Data Science team. The candidate should have a strong foundation in Python programming and basic knowledge of SQL for data querying. Understanding of core machine learning concepts including supervised and unsupervised learning, model evaluation metrics, and basic statistical concepts like hypothesis testing and probability is expected. Familiarity with libraries such as pandas, numpy, scikit-learn, and matplotlib is required. The candidate should have completed at least one end-to-end data science project demonstrating ability to collect, clean, analyze, and model data. Knowledge of Jupyter notebooks and Git for version control is expected. Good analytical thinking and problem-solving skills are more important than years of experience. Exposure to deep learning basics or NLP is a plus but not mandatory. Strong willingness to learn and grow in a fast-paced environment is essential.
Qualifications:

B.Tech / B.Sc in Computer Science, Statistics, Mathematics, or related field
0-1 years of experience
Final year students with strong projects are welcome to apply"""
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
