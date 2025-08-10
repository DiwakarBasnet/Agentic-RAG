import fitz
from pathlib import Path
from PIL import Image
import io
from unstructured.partition.pdf import partition_pdf

def identify_potential_table_pages(pdf_path: str):
    try:
        elements = partition_pdf(pdf_path)
        pages = {el.metadata.page_number for el in elements if el.category in ["Table", "UncategorizedText"]}
        return sorted(pages)
    except:
        doc = fitz.open(pdf_path)
        return list(range(1, len(doc) + 1))

def pdf_page_to_image(pdf_path: str, page_number: int, dpi: int = 150) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    doc.close()
    return image
