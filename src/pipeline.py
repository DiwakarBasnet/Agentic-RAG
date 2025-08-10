from src.processing.pdf_utils import identify_potential_table_pages, pdf_page_to_image
from src.processing.table_extraction import extract_tables_from_pages, save_dataframes_to_files
from src.processing.summarization import BLIPSummarizer
from src.models.metadata import create_simplified_metadata, create_summary_embeddings
from src.database.weaviate_utils import setup_weaviate_client, create_schema, store_in_weaviate
from src.config.config import OUTPUT_DIR, EMBEDDING_MODEL, BLIP_MODEL, WEAVIATE_URL, WEAVIATE_API_KEY
from typing import Dict

_PROMPT_SNIPPET = "Analyze this PDF page image and provide a detailed summary"

def _sanitize_summary(summary: str) -> str:
    # Remove accidental prompt echoes if present, and trim whitespace.
    if not summary:
        return summary
    # Heuristic: if our prompt snippet appears in the summary, strip everything up to and including it.
    if _PROMPT_SNIPPET in summary:
        idx = summary.find(_PROMPT_SNIPPET)
        cleaned = summary[:idx] + summary[idx + len(_PROMPT_SNIPPET):]
        return cleaned.strip()
    return summary.strip()

def process_pdf(pdf_path: str):
    pages = identify_potential_table_pages(pdf_path)
    if not pages:
        return []

    tables = extract_tables_from_pages(pdf_path, pages)
    if not tables:
        return []

    saved_files = save_dataframes_to_files(tables, pdf_path, OUTPUT_DIR)

    summarizer = BLIPSummarizer(BLIP_MODEL)

    # Build summaries only for pages we need and sanitize them
    page_summaries: Dict[int, str] = {}
    for p in pages:
        img = pdf_page_to_image(pdf_path, p)
        raw_summary = summarizer.summarize_financial_page(img)
        page_summaries[p] = _sanitize_summary(raw_summary)

    metadata = create_simplified_metadata(tables, pdf_path, saved_files, page_summaries)
    metadata = create_summary_embeddings(metadata, EMBEDDING_MODEL)

    client = setup_weaviate_client(WEAVIATE_URL, WEAVIATE_API_KEY)
    create_schema(client)
    store_in_weaviate(client, metadata)
    client.close()

    return [m["table_id"] for m in metadata]
