from typing import TypedDict, List, Dict, Any, Optional

class HiringState(TypedDict):
    job_title: str
    job_description_raw: str
    job_description_structured: Optional[Dict[str, Any]]
    candidates: List[Dict[str, Any]]
    selected_candidate: Optional[Dict[str, Any]]
    conversation_history: List[Any]
    current_suggestions: List[str]
    filters_applied: Dict[str, Any]
    feedback_cache: Dict[str, str]
    total_results: int
