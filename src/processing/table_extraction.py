import camelot
import os
import pandas as pd
from pathlib import Path

def extract_tables_from_pages(pdf_path, page_numbers):
    tables = []
    pages_str = ",".join(map(str, page_numbers))

    for flavor in ["lattice", "stream"]:
        try:
            extracted = camelot.read_pdf(pdf_path, pages=pages_str, flavor=flavor, suppress_stdout=True)
            for i, table in enumerate(extracted):
                df = table.df.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
                if not df.empty:
                    tables.append({
                        "dataframe": df,
                        "page_number": table.page,
                        "extraction_method": f"camelot_{flavor}",
                        "table_index": i
                    })
        except:
            pass
    return tables

def save_dataframes_to_files(tables, pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    pdf_name = Path(pdf_path).stem

    for i, tbl in enumerate(tables):
        file_path = os.path.join(output_dir, f"{pdf_name}_page{tbl['page_number']}_table{i+1}_{tbl['extraction_method']}.csv")
        tbl["dataframe"].to_csv(file_path, index=False)
        saved_files.append(file_path)
    return saved_files
