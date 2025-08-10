import warnings
from src.pipeline import process_pdf, SimpleRAG

warnings.filterwarnings("ignore")

def main():
    # Step 1: Process PDF & store in Weaviate
    pdf_path = "data/inputs/sample_input.pdf"
    process_pdf(pdf_path)

    # Step 2: Query data with RAG
    rag = SimpleRAG()
    try:
        question = "What expenses are mentioned in the documents"
        result = rag.answer_question(question, top_k=2, show_thinking=False)

        print("\nQuestion:", question)
        print("Answer:", result['answer'])
    finally:
        rag.close()

if __name__ == "__main__":
    main()
