from langgraph.graph import StateGraph, START, END
from src.state import HiringState
from src.agents.jd_agent import jd_understanding
from src.vectorstore import query_resumes
from src.scoring import compute_breakdown
from src.agents.feedback_agent import generate_feedback

def parse_jd_node(state: HiringState):
    raw_jd = state.get("job_description_raw", "")
    structured = jd_understanding(raw_jd)
    return {"job_description_structured": structured, "job_title": structured.get("job_title", "Unknown Role")}

def retrieve_and_score_node(state: HiringState):
    jd_raw = state.get("job_description_raw", "")
    filters = state.get("filters_applied", {})
    where = {}
    if filters.get("Category") and filters["Category"] != "All":
        where["Category"] = filters["Category"]
        
    if len(where) > 1:
        where_clause = {"$and": [{k: v} for k, v in where.items()]}
    elif len(where) == 1:
        where_clause = where
    else:
        where_clause = None

    # Calculate JD and facet embeddings ONCE to save time
    from src.vectorstore import get_embed_model
    from src.scoring import extract_skills_from_jd, extract_experience_from_jd, extract_education_from_jd
    model = get_embed_model()
    
    jd_lower = jd_raw.lower()
    skill_kws = extract_skills_from_jd(jd_lower)
    exp_kws = extract_experience_from_jd(jd_lower)
    edu_kws = extract_education_from_jd(jd_lower)
    
    # BGE models need the prefix on queries
    q_prefix = "Represent this sentence: "
    jd_emb = model.encode([q_prefix + jd_raw])[0]
    facet_embs = model.encode([
        q_prefix + (skill_kws or jd_raw),
        q_prefix + (exp_kws or jd_raw),
        q_prefix + (edu_kws or jd_raw)
    ])

    # retrieval (fetch up to 100 candidates)
    candidates = query_resumes(jd_raw, n_results=100, threshold=0.0, where=where_clause)
    
    # scoring
    processed = []
    for c in candidates:
        doc_emb = c.get("embedding")
        if doc_emb is None:
            continue
            
        bd = compute_breakdown(jd_emb, facet_embs, c["metadata"], doc_emb, c["score"])
        c["breakdown"] = bd
        # Remove embedding so it doesn't bloat session_state memory
        del c["embedding"]
        processed.append(c)
    
    return {"candidates": processed, "total_results": len(processed)}

def feedback_node(state: HiringState):
    selected = state.get("selected_candidate")
    parsed_jd = state.get("job_description_structured")
    cache = state.get("feedback_cache", {})
    
    if selected and parsed_jd:
        c_id = selected["id"]
        if c_id not in cache:
            try:
                fb = generate_feedback(selected, parsed_jd)
                cache[c_id] = fb
            except Exception as e:
                cache[c_id] = f"Error generating feedback: {e}"
            
    return {"feedback_cache": cache}

from src.agents.conversation_agent import handle_conversation

def chat_node(state: HiringState):
    return handle_conversation(state)

# Build graph
workflow = StateGraph(HiringState)
workflow.add_node("parse_jd", parse_jd_node)
workflow.add_node("retrieve_and_score", retrieve_and_score_node)
workflow.add_node("feedback", feedback_node)
workflow.add_node("chat", chat_node)

workflow.add_edge(START, "parse_jd")
workflow.add_edge("parse_jd", "retrieve_and_score")

def should_generate_feedback(state: HiringState):
    # This is a bit tricky, the graph is invoked on specific actions.
    # We can just let app.py call the specific nodes manually, or route based on state.
    # If there's a new chat message (len history is odd), route to chat.
    hist = state.get("conversation_history", [])
    if len(hist) > 0 and hist[-1]["role"] == "user":
        return "chat"
    if state.get("selected_candidate"):
        return "feedback"
    return END

workflow.add_conditional_edges("retrieve_and_score", should_generate_feedback)
workflow.add_edge("feedback", END)
workflow.add_edge("chat", END)

app_graph = workflow.compile()
