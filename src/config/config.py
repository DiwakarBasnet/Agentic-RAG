import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Paths
OUTPUT_DIR = Path("data/outputs/table_data")

# Weaviate configuration
WEAVIATE_URL = os.getenv('WEAVIATE_URL')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')
WEAVIATE_CLASS = os.getenv('WEAVIATE_CLASS')

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MULTI_MODAL_MODEL = "Salesforce/blip-image-captioning-large"
LLM_MODEL = "Gensyn/Qwen2.5-0.5B-Instruct"
HF_TOKEN = os.getenv('HF_TOKEN')

