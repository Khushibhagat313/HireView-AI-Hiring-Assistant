"""
Utility to look up the full resume text from the original dataset by ResumeID.
This lets us bypass the 3000-char ChromaDB truncation for detailed AI Q&A.
"""
import pandas as pd
import os

_df = None  # cache

def _load_df():
    global _df
    if _df is None:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "resumes_dataset.jsonl")
        path = os.path.normpath(path)
        if os.path.exists(path):
            _df = pd.read_json(path, lines=True)
        else:
            _df = pd.DataFrame()
    return _df

def get_full_resume_text(resume_id: str) -> str:
    """Return the full resume text from the original dataset for a given ResumeID."""
    df = _load_df()
    if df.empty:
        return ""
    row = df[df["ResumeID"] == resume_id]
    if row.empty:
        return ""
    text = str(row.iloc[0].get("Text", ""))
    category = str(row.iloc[0].get("Category", ""))
    return f"Role: {category}\n{text}"
