import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --- LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "groq"
GEMINI_MODEL = "gemini-2.0-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

ARXIV_RAW_PATH = os.path.join(DATA_DIR, "arxiv-metadata-oai-snapshot.json")
PAPERS_JSONL_PATH = os.path.join(DATA_DIR, "arxiv_cs_papers.jsonl")
BM25_INDEX_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")

# --- Models ---
DENSE_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- Retrieval hyperparameters ---
BM25_TOP_K = 100
DENSE_TOP_K = 100
RRF_K = 60
RERANK_TOP_N = 10
RERANK_CANDIDATES = 30  # rerank top 30 to get best RERANK_TOP_N

# --- Server ---
API_HOST = "0.0.0.0"
API_PORT = 8000
