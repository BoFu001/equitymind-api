import os
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
SEC_FILINGS_DIR = os.path.join(DATA_DIR, "sec-edgar-filings")

# Chunking
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# API Keys
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
FINLIGHT_API_KEY = os.getenv("FINLIGHT_API_KEY")

# Batch sizes
PINECONE_BATCH_SIZE  = 100
EMBEDDING_BATCH_SIZE = 100