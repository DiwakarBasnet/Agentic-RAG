from src.processing.summarization import BLIPSummarizer
from src.processing.pdf_utils import pdf_page_to_image

model_name = "Salesforce/blip-image-captioning-large"
s = BLIPSummarizer(model_name)
img = pdf_page_to_image("pdf_input.pdf", 4)
summary = s.summarize_financial_page(img)
print("==== SUMMARY (repr) ====")
print(repr(summary[:500]))
print("=========================")
