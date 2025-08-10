from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass
class TableResult:
    table_id: str
    csv_file_path: str
    page_summary: str
    page_number: int
    source_file: str
    certainty: float

@dataclass
class RAGAnswer:
    thinking: str
    answer: str
    raw_answer: str

def create_simplified_metadata(tables, pdf_path, saved_files, page_summaries):
    metadata_list = []
    for i, tbl in enumerate(tables):
        metadata_list.append({
            "table_id": f"{Path(pdf_path).stem}_page{tbl['page_number']}_table{i+1}_{tbl['extraction_method']}",
            "csv_file_path": saved_files[i],
            "page_summary": page_summaries.get(tbl['page_number'], "").strip(),
            "page_number": tbl['page_number'],
            "source_file": Path(pdf_path).name,
            "created_at": pd.Timestamp.now().isoformat()
        })
    return metadata_list

def create_summary_embeddings(metadata_list, model_name):
    model = SentenceTransformer(model_name)
    for m in metadata_list:
        m["embedding"] = model.encode(m["page_summary"]).tolist()
        m["embedding_model"] = model_name
    return metadata_list
