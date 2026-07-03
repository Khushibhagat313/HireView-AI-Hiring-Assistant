import pandas as pd

def run_eda():
    df = pd.read_json("data/resumes_dataset.jsonl", lines=True)
    print(f"Total Rows: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
    
    print("\nSource split:")
    print(df['Source'].value_counts())
    
    skills_ph = "Python, SQL, Git, Linux"
    edu_ph = "Computer Science degree"
    email_ph = "contact@email.com"
    
    skills_pct = (df['Skills'] == skills_ph).mean() * 100
    edu_pct = (df['Education'] == edu_ph).mean() * 100
    email_pct = (df['Email'] == email_ph).mean() * 100
    
    print(f"\nPlaceholder Percentages:")
    print(f"Skills == '{skills_ph}': {skills_pct:.2f}%")
    print(f"Education == '{edu_ph}': {edu_pct:.2f}%")
    print(f"Email == '{email_ph}': {email_pct:.2f}%")
    
    print("\nSource distribution for placeholder Skills:")
    print(df[df['Skills'] == skills_ph]['Source'].value_counts())
    
    print("\nWhy Skills/Education can't be trusted for ResumeAtlas rows:")
    print("Because ResumeAtlas data appears to use hardcoded placeholder strings for these fields instead of the candidate's actual extracted skills and education.")

if __name__ == "__main__":
    run_eda()
