import os

# Paths
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
SEC_FILINGS_DIR = os.path.join(DATA_DIR, "sec-edgar-filings")

# Chunking (tunable for experiments)
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200