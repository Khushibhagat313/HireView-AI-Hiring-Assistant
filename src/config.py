import os
from dotenv import load_dotenv

load_dotenv()

# BGE-base-en-v1.5: purpose-built retrieval model, free, local, top of MTEB leaderboard
# Requires prepending "Represent this sentence: " to queries for best results
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
CHROMA_DIR = "chroma_db"
COLLECTION = "resumes"
THRESHOLD = 0.40  # realistic threshold for mpnet model

# LLM Constants
LLM_MODEL = "qwen/qwen3-32b"
LLM_TEMPERATURE = 0.2
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
