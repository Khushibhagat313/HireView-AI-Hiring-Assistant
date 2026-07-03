import re
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from src.config import LLM_MODEL, LLM_TEMPERATURE, GROQ_API_KEY
from src.prompts import JD_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class JDRequirements(BaseModel):
    job_title: str = Field(description="The job title")
    required_skills: List[str] = Field(description="List of required technical and soft skills")
    experience_requirement: str = Field(description="Years of experience required or level")
    education_requirement: str = Field(description="Education requirements")
    responsibilities: str = Field(description="Main responsibilities context")

def get_llm():
    return ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=GROQ_API_KEY
    )

def jd_understanding(raw_text_or_fields: str | Dict[str, str]) -> dict:
    llm = get_llm()
    
    # If it's a dict (Structured mode), format it nicely
    if isinstance(raw_text_or_fields, dict):
        content = "\n".join([f"{k}: {v}" for k, v in raw_text_or_fields.items() if v])
    else:
        content = raw_text_or_fields
        
    messages = [
        SystemMessage(content=JD_SYSTEM_PROMPT),
        HumanMessage(content=f"Here is the job description input:\n{content}\n\nPlease extract the requirements.")
    ]
    
    try:
        structured_llm = llm.with_structured_output(JDRequirements)
        result = structured_llm.invoke(messages)
        return result.model_dump()
    except Exception as e:
        print(f"Structured output failed, falling back to manual parsing: {e}")
        return {
            "job_title": "Parsed Title",
            "required_skills": [],
            "experience_requirement": "Error parsing",
            "education_requirement": "Error parsing",
            "responsibilities": "Error parsing"
        }
