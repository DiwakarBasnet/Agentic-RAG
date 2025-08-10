import fitz
import camelot
from typing import List, Dict
from unstructured.partition.pdf import partition_pdf

def identify_potential_table_pages(pdf_path: str) -> List[int]:
    """Identify potential table pages using unstructured."""
    try:
        elements = partition_pdf(pdf_path)
        pages = {
            el.metadata.page_number
            for el in elements
            if getattr(el, "category", None) in {"Table", "UncategorizedText"}
            and hasattr(el, "metadata")
        }
        return sorted(pages)
    except Exception:
        with fitz.open(pdf_path) as doc:
            return list(range(1, len(doc) + 1))

def extract_tables_from_pages(pdf_path: str, page_numbers: List[int]) -> List[Dict]:
    """Extract tables from specified PDF pages using Camelot (both flavors)."""
    if not page_numbers:
        return []

    pages_str = ",".join(map(str, page_numbers))
    all_tables = []

    for flavor in ["lattice", "stream"]:
        try:
            tables = camelot.read_pdf(pdf_path, pages=pages_str, flavor=flavor, suppress_stdout=True)
            for i, table in enumerate(tables):
                df = table.df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
                if df.empty:
                    continue
                table_info = {
                    "dataframe": df,
                    "page_number": table.page,
                    "extraction_method": f"camelot_{flavor}",
                    "table_index": i,
                    "parsing_report": getattr(table, "parsing_report", None),
                }
                if not any(
                    t["page_number"] == table.page and t["dataframe"].shape == df.shape
                    for t in all_tables
                ):
                    all_tables.append(table_info)
        except Exception:
            continue
    return all_tables
