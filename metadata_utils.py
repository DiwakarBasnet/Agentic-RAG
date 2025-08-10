# metadata_utils.py
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer

def save_dataframes_to_files(table_info_list: List[Dict], pdf_path: str, output_dir: Path) -> List[str]:
    """Save DataFrames to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    pdf_name = Path(pdf_path).stem

    for i, table_info in enumerate(table_info_list):
        filename = f"{pdf_name}_page{table_info['page_number']}_table{i+1}_{table_info['extraction_method']}.csv"
        file_path = output_dir / filename
        table_info["dataframe"].to_csv(file_path, index=False)
        saved_files.append(str(file_path))
    return saved_files

def create_embeddings(metadata_list: List[Dict], model_name: str) -> List[Dict]:
    """Add embeddings to metadata."""
    model = SentenceTransformer(model_name)
    for metadata in metadata_list:
        text = f"Table from {metadata['source_file']} page {metadata['page_number']}. {metadata['sample_text']}"
        metadata["embedding"] = model.encode(text).tolist()
        metadata["embedding_model"] = model_name
    return metadata_list

def get_page_title(page_text: str) -> str:
    """Extract the title from page text (simple heuristic: first non-empty line)."""
    lines = [line.strip() for line in page_text.split("\n") if line.strip()]
    return lines[0] if lines else "Untitled"

def create_table_metadata(tables, pdf_path, saved_files, page_text_map):
    metadata_list = []
    for i, table in enumerate(tables):
        page_num = table["page_number"]
        page_text = page_text_map.get(page_num, "")
        page_title = get_page_title(page_text)

        # Generate a sample text snippet from the table
        df = table["dataframe"]
        sample_text = df.head(3).to_string(index=False) if not df.empty else ""

        metadata = {
            "table_id": f"{pdf_path}_table_{i}",
            "source_file": pdf_path,
            "page_number": page_num,
            "csv_file_path": saved_files[i],
            "shape": str(df.shape),
            "extraction_method": table["extraction_method"],
            "page_title": page_title,
            "sample_text": sample_text 
        }
        metadata_list.append(metadata)
    return metadata_list
