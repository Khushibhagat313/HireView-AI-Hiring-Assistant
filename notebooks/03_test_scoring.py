from src.vectorstore import query_resumes
from src.scoring import compute_breakdown

def screen(jd):
    print(f"Screening for JD: {jd}")
    candidates = query_resumes(jd, n_results=5, threshold=0.0) # Lower threshold for testing
    for i, c in enumerate(candidates):
        bd = compute_breakdown(jd, c["metadata"], c["document"])
        name = c["metadata"].get("Name", "Unknown")
        source = c["metadata"].get("Source", "Unknown")
        print(f"\n#{i+1} {name} ({source}) - Overall Vector Score: {c['score_pct']}%")
        print(f"  Skills: {bd['skills_score']}% [{bd['skills_tag']}]")
        print(f"  Experience: {bd['exp_score']}% [{bd['exp_tag']}]")
        print(f"  Education: {bd['edu_score']}% [{bd['edu_tag']}]")
        print(f"  Composite: {bd['composite_score']}%")

if __name__ == "__main__":
    screen("Senior Python developer, 5y, ML, AWS")
