from config import OUTPUT_DIR, EMBEDDING_MODEL
from table_extraction import identify_potential_table_pages, extract_tables_from_pages
from metadata_utils import save_dataframes_to_files, create_table_metadata, create_embeddings
from weaviate_utils import setup_weaviate_client, create_weaviate_schema, store_metadata_in_weaviate
import pdfplumber

def process_pdf_pipeline(pdf_path: str):
    pages = identify_potential_table_pages(pdf_path)
    tables = extract_tables_from_pages(pdf_path, pages)
    if not tables:
        return []
    saved_files = save_dataframes_to_files(tables, pdf_path, OUTPUT_DIR)
    page_text_map = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in pages:
            page_text_map[page_num] = pdf.pages[page_num - 1].extract_text() or ""
    metadata = create_table_metadata(tables, pdf_path, saved_files, page_text_map)
    metadata_with_embeddings = create_embeddings(metadata, EMBEDDING_MODEL)
    client = setup_weaviate_client()
    create_weaviate_schema(client)
    ids = store_metadata_in_weaviate(client, metadata_with_embeddings)
    client.close()
    
    return ids
