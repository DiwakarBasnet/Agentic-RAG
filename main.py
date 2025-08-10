import sys
from pipeline import process_pdf_pipeline
from weaviate_utils import setup_weaviate_client, search_similar_tables

def print_usage():
    print("""
Usage:
  python main.py process <pdf_path>       Process a PDF and store tables in Weaviate
  python main.py search "<query_text>"    Search for similar tables in Weaviate

Examples:
  python main.py process pdf_input.pdf
  python main.py search "financial data revenue"
""")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "process":
        if len(sys.argv) < 3:
            print("Error: Missing <pdf_path> for processing.\n")
            print_usage()
            sys.exit(1)
        pdf_path = sys.argv[2]
        ids = process_pdf_pipeline(pdf_path)
        print(f"Stored {len(ids)} tables in Weaviate.")

    elif command == "search":
        if len(sys.argv) < 3:
            print("Error: Missing <query_text> for search.\n")
            print_usage()
            sys.exit(1)
        query_text = sys.argv[2]
        client = setup_weaviate_client()
        results = search_similar_tables(client, query_text)
        client.close()

        if not results:
            print("No results found.")
        else:
            print(f"Found {len(results)} matching tables:\n")
            for res in results:
                props = res["properties"]
                meta = res["metadata"]
                print(f"- Table ID: {props['table_id']}")
                print(f"  Page: {props['page_number']}, Method: {props['extraction_method']}")
                print(f"  CSV Path: {props['csv_file_path']}")
                print(f"  Shape: {props['shape']}")
                print(f"  Certainty: {meta.get('certainty'):.3f} | Distance: {meta.get('distance'):.3f}")
                print()

    else:
        print(f"Unknown command: {command}\n")
        print_usage()
        sys.exit(1)
