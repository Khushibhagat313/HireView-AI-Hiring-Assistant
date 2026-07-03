import re
from langchain_groq import ChatGroq
from src.config import LLM_MODEL, LLM_TEMPERATURE, GROQ_API_KEY
from src.prompts import FEEDBACK_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks that Qwen3 emits as reasoning traces."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate_feedback(candidate: dict, jd: dict) -> str:
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=GROQ_API_KEY
    )
    
    resume_text = candidate.get("document", "")[:3000]
    score_pct = candidate.get("score_pct", 0)
    name = candidate.get("metadata", {}).get("Name", "Candidate")
    
    jd_formatted = "\n".join([f"{k}: {v}" for k, v in jd.items()])
    
    prompt = f"""Candidate Name: {name}
Match Score: {score_pct}%

Resume (excerpt):
{resume_text}

Job Requirements:
{jd_formatted}

Please generate structured feedback as requested in your system prompt."""

    messages = [
        SystemMessage(content=FEEDBACK_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    return strip_think_tags(response.content)
