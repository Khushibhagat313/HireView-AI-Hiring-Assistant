import re
import numpy as np
from src.vectorstore import get_embed_model

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def compute_breakdown(jd_emb, facet_embs, metadata, doc_emb, overall):
    """
    Compute a detailed score breakdown using pre-computed embeddings.
    jd_emb: 1D array
    facet_embs: list of three 1D arrays [skills, exp, edu]
    doc_emb: 1D array
    overall: float
    """
    # Candidate document embedding
    s_score = cosine_similarity(facet_embs[0], doc_emb)
    x_score = cosine_similarity(facet_embs[1], doc_emb)
    e_score = cosine_similarity(facet_embs[2], doc_emb)

    # Weighted composite
    composite = 0.45 * s_score + 0.35 * x_score + 0.20 * e_score

    return {
        "skills_score": int(round(max(0, s_score) * 100)),
        "skills_tag": "context-matched",
        "exp_score": int(round(max(0, x_score) * 100)),
        "exp_tag": "context-matched",
        "edu_score": int(round(max(0, e_score) * 100)),
        "edu_tag": "context-matched",
        "composite_score": int(round(max(0, composite) * 100)),
        "overall_similarity": int(round(max(0, overall) * 100))
    }

def extract_skills_from_jd(jd_lower: str) -> str:
    """Pull skill/technology sentences from JD."""
    skill_patterns = [
        r'(proficien[cy].*?[.\n])', r'(experience with.*?[.\n])', r'(knowledge of.*?[.\n])',
        r'(familiarity with.*?[.\n])', r'(skilled in.*?[.\n])', r'(expertise in.*?[.\n])',
        r'(must have.*?[.\n])', r'(required.*?skill.*?[.\n])',
    ]
    sentences = []
    for pat in skill_patterns:
        for m in re.finditer(pat, jd_lower, re.IGNORECASE):
            sentences.append(m.group(0).strip())
    return " ".join(sentences[:6]) if sentences else ""

def extract_experience_from_jd(jd_lower: str) -> str:
    """Pull experience-related sentences from JD."""
    exp_patterns = [
        r'(\d+\+?\s*years?.*?[.\n])', r'(experience.*?[.\n])', r'(background in.*?[.\n])',
        r'(previous.*?[.\n])', r'(worked.*?[.\n])', r'(delivered.*?[.\n])',
    ]
    sentences = []
    for pat in exp_patterns:
        for m in re.finditer(pat, jd_lower, re.IGNORECASE):
            sentences.append(m.group(0).strip())
    return " ".join(sentences[:5]) if sentences else ""

def extract_education_from_jd(jd_lower: str) -> str:
    """Pull education-related sentences from JD."""
    edu_patterns = [
        r'(bachelor.*?[.\n])', r'(master.*?[.\n])', r'(degree.*?[.\n])',
        r'(b\.tech.*?[.\n])', r'(b\.sc.*?[.\n])', r'(phd.*?[.\n])',
        r'(qualification.*?[.\n])',
    ]
    sentences = []
    for pat in edu_patterns:
        for m in re.finditer(pat, jd_lower, re.IGNORECASE):
            sentences.append(m.group(0).strip())
    return " ".join(sentences[:3]) if sentences else ""
