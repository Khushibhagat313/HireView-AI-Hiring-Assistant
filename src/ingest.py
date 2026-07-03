import pandas as pd
from tqdm import tqdm
from src.vectorstore import get_chroma_client, get_collection, add_documents

def run_ingestion(batch_size=100):
    print("Loading dataset...")
    df = pd.read_json("data/resumes_dataset.jsonl", lines=True)
    total_docs = len(df)
    print(f"Total resumes to process: {total_docs}")
    
    # Check if already ingested
    client = get_chroma_client()
    collection = get_collection(client)
    
    if collection.count() >= total_docs:
        print(f"Collection already has {collection.count()} docs. Skipping ingestion.")
        return

    docs = []
    metas = []
    ids = []
    
    for i, row in tqdm(df.iterrows(), total=total_docs, desc="Ingesting"):
        resume_id = str(row.get("ResumeID", f"resume_{i}"))
        
        # Build embedded text
        embedded_text = f"""Category: {row.get('Category', '')}
Summary: {row.get('Summary', '')}
Skills: {row.get('Skills', '')}
Experience: {row.get('Experience', '')}
Education: {row.get('Education', '')}
Text: {row.get('Text', '')}"""
        
        # Metadata
        metadata = {
            "ResumeID": resume_id,
            "Name": row.get("Name", ""),
            "Category": row.get("Category", ""),
            "Source": row.get("Source", ""),
            "Email": row.get("Email", ""),
            "Phone": row.get("Phone", ""),
            "Location": row.get("Location", "")
        }
        
        docs.append(embedded_text)
        metas.append(metadata)
        ids.append(resume_id)
        
        if len(docs) >= batch_size:
            add_documents(docs, metas, ids)
            docs = []
            metas = []
            ids = []
            
    if len(docs) > 0:
        add_documents(docs, metas, ids)
        
    print(f"Ingestion complete. Total in ChromaDB: {collection.count()}")

if __name__ == "__main__":
    run_ingestion()
