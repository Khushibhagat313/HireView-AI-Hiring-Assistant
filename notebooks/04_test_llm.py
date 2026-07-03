import sys
from dotenv import load_dotenv
load_dotenv()

from src.agents.jd_agent import jd_understanding
from src.agents.feedback_agent import generate_feedback

def test():
    jd = "We need a Senior Python developer with 5 years experience, ML and AWS skills. Computer Science degree."
    print("Testing JD Agent...")
    try:
        res = jd_understanding(jd)
        print(res)
    except Exception as e:
        print(f"JD Agent Failed: {e}")
        return
    
    print("\nTesting Feedback Agent...")
    c = {
        "document": "Python developer with 6 years experience. Expert in ML and AWS. B.Tech in CS.",
        "score_pct": 95
    }
    try:
        feedback = generate_feedback(c, res)
        print(feedback)
    except Exception as e:
        print(f"Feedback Agent Failed: {e}")

if __name__ == "__main__":
    test()
