import re
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.config import LLM_MODEL, LLM_TEMPERATURE, GROQ_API_KEY
from src.dataset_utils import get_full_resume_text


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks that Qwen3 emits as reasoning traces."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def handle_conversation(state):
    history = state.get("conversation_history", [])
    if not history:
        return {"conversation_history": []}

    # Build context
    jd = state.get("job_description_structured", {})
    selected = state.get("selected_candidate")
    candidates = state.get("candidates", [])

    context = f"Job Requirements:\n{jd}\n\n"

    if selected:
        rid = selected.get("id", "")
        # Try to get full text from the original dataset (bypasses 3000-char ChromaDB limit)
        full_text = get_full_resume_text(rid)
        if not full_text:
            # Fall back to ChromaDB stored text
            full_text = selected.get("document", "")
        full_text = full_text[full_text.find('\n')+1:] if '\n' in full_text else full_text
        context += (
            f"Selected candidate: {selected['metadata'].get('Name')} "
            f"({selected['score_pct']}% match, Category: {selected['metadata'].get('Category')})\n"
            f"Full Resume:\n{full_text[:5000]}\n\n"
        )

    if candidates:
        context += f"All candidates ({len(candidates)} total, ranked by match %):\n"
        for i, c in enumerate(candidates[:20], 1):
            name = c['metadata'].get('Name', '?')
            cat  = c['metadata'].get('Category', '?')
            pct  = c['score_pct']
            if i <= 5:
                rid = c.get("id", "")
                full_text = get_full_resume_text(rid)
                if not full_text:
                    full_text = c.get("document", "")
                full_text = full_text[full_text.find('\n')+1:] if '\n' in full_text else full_text
                context += f"\n--- Candidate {i}: {name} | {cat} | {pct}% ---\n{full_text[:3000]}\n"
            else:
                context += f"  {i}. {name} — {cat} — {pct}%\n"

    system_prompt = f"""You are a concise AI hiring assistant. Help the recruiter make decisions fast.

Rules:
- Answer in 5-10 lines maximum. Be direct and specific.
- Use bullet points or short tables for comparisons. 
- NO long paragraphs. NO preamble.
- Never repeat the question back.
- Base answers ONLY on the context below. Do not invent details.

Context:
{context}"""

    # DEBUG: Print the first 1000 chars of the context to terminal to verify
    print("--- DEBUG CONTEXT START ---")
    print(system_prompt[:1000])
    print(f"Total context size: {len(system_prompt)} chars")
    print("--- DEBUG CONTEXT END ---")

    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            # Pass clean version back to LLM too
            clean_content = strip_think_tags(msg["content"])
            messages.append(AIMessage(content=clean_content))

    try:
        # Standard initialization (no model_kwargs for reasoning_effort to avoid Pydantic ValidationError)
        llm = ChatGroq(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=GROQ_API_KEY
        )
        response = llm.invoke(messages)
        clean = strip_think_tags(response.content)
        new_msg = {"role": "assistant", "content": clean}
        return {"conversation_history": history + [new_msg]}
    except Exception as e:
        err = {"role": "assistant", "content": f"Error during generation: {e}"}
        return {"conversation_history": history + [err]}
