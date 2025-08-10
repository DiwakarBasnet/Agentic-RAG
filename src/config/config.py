from pathlib import Path

# Paths
OUTPUT_DIR = Path("data/outputs/table_data")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Weaviate configuration
WEAVIATE_URL = "https://6rbkz8vnqryn3yw5qtqkpg.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "U1U3TkVDalVKOGxZQmdicV9TT2xWeElSS0IvK2dKTlJLTk1zYXFoSDVDWmdtZDgxYWRJeVYrZHJpMUJVPV92MjAw"

# Class name for storing metadata
WEAVIATE_CLASS = "PDFTableMetadata"

# Multi modal model
BLIP_MODEL = "Salesforce/blip-image-captioning-large"
