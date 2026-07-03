# HireView - AI Hiring Assistant

HireView is an intelligent, AI-powered hiring assistant that simplifies and accelerates the recruitment process. By leveraging advanced natural language processing and semantic search, it helps recruiters find the perfect candidates from large pools of resumes based on unstructured job descriptions.

## 🚀 What Our Project Is

Finding the right candidate from hundreds of resumes is a tedious, time-consuming process. HireView solves this by bringing **Semantic Search** and **AI Chat** directly to the recruiter's workflow. Instead of keyword matching, HireView understands the *context* of a job description and scores candidates based on their actual skills, experience, and education. 

It provides an interactive interface where hiring managers can instantly upload resumes, search for candidates, view detailed AI-generated feedback, and even "chat" with an AI assistant to compare candidates.

---

## 📸 Interface & Workflow

### 1. Job Description & AI Assistant
To get started, you can either paste a full unstructured job description or use the structured form. The UI is clean and clearly separated into the Search parameters and the AI Assistant chat.
![Job Description & AI Assistant](screenshot_1.png)

### 2. Candidate Results & Details
Once you search, the middle panel displays a ranked list of candidates based on semantic match percentage. Clicking on a candidate opens the right panel with a detailed breakdown of their scores across Skills, Experience, and Education, along with a full Resume Preview.
![Candidate Results & Details](screenshot_2.png)

### 3. AI Feedback & Score Validation
HireView automatically generates highly targeted AI Feedback for the selected candidate. It breaks down their specific strengths for the role, identifies skill gaps, and validates exactly *why* they received their match score.
![AI Feedback](screenshot_3.png)

### 4. Chat with AI Recruiter & Quick Actions
Want to go deeper? The interactive Chat panel at the bottom right allows you to ask the AI questions like "What is his education background?" or "Compare the top 3 candidates." You can also use Quick Actions like "Skill Gap Analysis" or "Interview Questions" to get immediate, context-aware suggestions!
![Chat with AI Recruiter](screenshot_4.png)

---

## 🛠️ Features & Functionalities

*   **Semantic Resume Search:** Instead of relying on exact keyword matches, HireView understands the *intent* behind a job description using the `BAAI/bge-base-en-v1.5` embedding model.
*   **Flexible Inputs:** Paste a full text job description directly, or use a structured form.
*   **Intelligent AI Recruiter Chat:** A built-in chat interface powered by Groq and Qwen3 allows you to ask complex, comparative questions. The AI reads the *actual* resume texts to provide highly accurate, grounded answers.
*   **Dynamic Feedback:** View detailed score breakdowns. The AI analyzes the candidate's strengths and weaknesses relative to the role.
*   **Resume Uploads:** Upload new PDF resumes on the fly. They are instantly parsed, embedded, and added to the local database for immediate searching.
*   **Independent Scrolling UI:** The Results List and the Candidate Details panels scroll independently, ensuring the Chat input is always accessible at the bottom of the screen.

## 💻 Tech Stack
*   **Frontend:** Streamlit
*   **Embeddings & Database:** SentenceTransformers (`bge-base-en-v1.5`), ChromaDB
*   **LLM Integration:** LangChain, Groq API (Qwen3)
*   **Workflow Logic:** LangGraph

## 🔒 Security Note
**All API keys and local datasets are fully secured.** The `.env` file containing the `GROQ_API_KEY` is completely ignored by version control (via `.gitignore`), ensuring that no private credentials or raw candidate data are ever pushed to GitHub.
