"""
Re-ingestion script: wipes the ChromaDB collection and re-ingests all resumes.
Embeds ONLY the `Text` field (the actual resume content) to avoid contamination
from broken placeholder fields (Skills, Education) in the ResumeAtlas data.

Run with:
  $env:PYTHONPATH="."; uv run python src/reingest.py
"""
import pandas as pd
from tqdm import tqdm
from src.vectorstore import get_chroma_client, get_collection, get_embed_model, add_documents
import chromadb

DATASET_PATH = "data/resumes_dataset.jsonl"
BATCH_SIZE = 64

def clean_text(text: str) -> str:
    """Basic cleanup of OCR-noise text."""
    if not isinstance(text, str):
        return ""
    # Collapse excessive whitespace
    import re
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_embed_text(row) -> str:
    """
    Build the text to embed. Strategy:
    - For all rows: use `Text` field (full resume content - reliable across sources)
    - Prepend Category so the model understands job domain context
    - If Synthetic source, also prepend structured fields (they are real, not placeholder)
    """
    category = str(row.get("Category", "")).strip()
    source = str(row.get("Source", "")).strip()
    text = clean_text(str(row.get("Text", "")))

    # For Synthetic rows, structured fields are real - include them
    if source == "Synthetic":
        summary = clean_text(str(row.get("Summary", "")))
        skills = clean_text(str(row.get("Skills", "")))
        exp = clean_text(str(row.get("Experience", "")))
        edu = clean_text(str(row.get("Education", "")))
        return f"Role: {category}\nSkills: {skills}\nExperience: {exp}\nEducation: {edu}\n{summary}\n{text}"
    else:
        # ResumeAtlas: Skills/Education are fake placeholders, only Text is real
        return f"Role: {category}\n{text}"

def run_reingest():
    print("Loading dataset...")
    df = pd.read_json(DATASET_PATH, lines=True)
    total = len(df)
    print(f"Total resumes: {total}")

    # Wipe and recreate the collection
    client = get_chroma_client()
    try:
        client.delete_collection("resumes")
        print("Deleted old collection.")
    except Exception:
        print("No existing collection to delete.")
    collection = get_collection(client)

    print("Pre-loading embedding model...")
    model = get_embed_model()
    print(f"Model: {model}")

    docs, metas, ids = [], [], []
    skipped = 0

    for i, row in tqdm(df.iterrows(), total=total, desc="Building docs"):
        resume_id = str(row.get("ResumeID", f"resume_{i}"))
        embed_text = build_embed_text(row)

        if not embed_text.strip() or len(embed_text) < 50:
            skipped += 1
            continue

        # Truncate to avoid token limits (mpnet max is 384 tokens ~ 1500 chars)
        embed_text = embed_text[:3000]

        meta = {
            "ResumeID": resume_id,
            "Name": str(row.get("Name", "Unknown")),
            "Category": str(row.get("Category", "Unknown")),
            "Source": str(row.get("Source", "Unknown")),
            "Email": str(row.get("Email", "")),
            "Location": str(row.get("Location", "")),
        }

        docs.append(embed_text)
        metas.append(meta)
        ids.append(resume_id)

        if len(docs) >= BATCH_SIZE:
            add_documents(docs, metas, ids)
            docs, metas, ids = [], [], []

    if docs:
        add_documents(docs, metas, ids)

    final_count = collection.count()
    print(f"\n✅ Ingestion complete!")
    print(f"   Ingested: {final_count} | Skipped (empty): {skipped}")

if __name__ == "__main__":
    run_reingest()
