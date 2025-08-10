import fitz  # PyMuPDF
import camelot
import pandas as pd
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from weaviate.classes.init import Auth
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import uuid
import os
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
import base64
from PIL import Image
import io
import requests
from openai import OpenAI
import time


def identify_potential_table_pages(pdf_path: str) -> List[int]:
    """
    Identify potential table pages using unstructured library.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        List[int]: List of page numbers that potentially contain tables
    """
    try:
        # Partition PDF to identify elements
        elements = partition_pdf(pdf_path)
        
        # Filter for table and uncategorized text elements
        table_elements = [el for el in elements if el.category == "Table"]
        uncategorized_text_elements = [el for el in elements if el.category == "UncategorizedText"]
        
        # Determine potential table pages
        potential_table_pages = set()
        for el in table_elements + uncategorized_text_elements:
            if hasattr(el, 'metadata') and hasattr(el.metadata, 'page_number'):
                potential_table_pages.add(el.metadata.page_number)
        
        potential_pages = sorted(list(potential_table_pages))
        print(f"Potential table pages identified: {potential_pages}")
        
        return potential_pages
        
    except Exception as e:
        print(f"Error identifying potential table pages: {e}")
        # Fallback to all pages
        doc = fitz.open(pdf_path)
        all_pages = list(range(1, len(doc) + 1))
        doc.close()
        return all_pages


def extract_tables_from_pages(pdf_path: str, page_numbers: List[int]) -> List[Dict]:
    """
    Extract tables from specified PDF pages using both Camelot flavors.
    
    Args:
        pdf_path (str): Path to the PDF file
        page_numbers (List[int]): List of page numbers to process
        
    Returns:
        List[Dict]: List of dictionaries containing DataFrames and metadata
    """
    all_tables = []
    
    if not page_numbers:
        print("No pages to process")
        return all_tables
    
    # Convert page numbers to comma-separated string
    pages_str = ",".join(map(str, page_numbers))
    print(f"Extracting tables from pages: {pages_str}")
    
    # Try lattice flavor first
    try:
        print("  Trying 'lattice' flavor...")
        tables_lattice = camelot.read_pdf(pdf_path, pages=pages_str, flavor='lattice', suppress_stdout=True)
        print(f"  Camelot (lattice) found {tables_lattice.n} tables.")
        
        for i, table in enumerate(tables_lattice):
            df = table.df
            # Clean up the DataFrame
            df = df.dropna(how='all').dropna(axis=1, how='all')
            df = df.reset_index(drop=True)
            
            if not df.empty:
                table_info = {
                    'dataframe': df,
                    'page_number': table.page,
                    'extraction_method': 'camelot_lattice',
                    'table_index': i,
                    'parsing_report': table.parsing_report if hasattr(table, 'parsing_report') else None
                }
                all_tables.append(table_info)
                print(f"    Added table {i+1} from page {table.page} (lattice)")
    
    except Exception as e:
        print(f"  Error with lattice flavor: {e}")
    
    # Try stream flavor
    try:
        print("  Trying 'stream' flavor...")
        tables_stream = camelot.read_pdf(pdf_path, pages=pages_str, flavor='stream', suppress_stdout=True)
        print(f"  Camelot (stream) found {tables_stream.n} tables.")
        
        for i, table in enumerate(tables_stream):
            df = table.df
            # Clean up the DataFrame
            df = df.dropna(how='all').dropna(axis=1, how='all')
            df = df.reset_index(drop=True)
            
            if not df.empty:
                # Check if we already have a similar table from lattice method
                is_duplicate = False
                for existing_table in all_tables:
                    if (existing_table['page_number'] == table.page and 
                        existing_table['dataframe'].shape == df.shape):
                        # Simple duplicate check based on shape and page
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    table_info = {
                        'dataframe': df,
                        'page_number': table.page,
                        'extraction_method': 'camelot_stream',
                        'table_index': i,
                        'parsing_report': table.parsing_report if hasattr(table, 'parsing_report') else None
                    }
                    all_tables.append(table_info)
                    print(f"    Added table {i+1} from page {table.page} (stream)")
                else:
                    print(f"    Skipped duplicate table {i+1} from page {table.page} (stream)")
    
    except Exception as e:
        print(f"  Error with stream flavor: {e}")
    
    if not all_tables:
        print("  No tables found with either flavor.")
    
    return all_tables


def save_dataframes_to_files(table_info_list: List[Dict], pdf_path: str, 
                           output_dir: str = "table_data") -> List[str]:
    """
    Save DataFrames to separate CSV files.
    
    Args:
        table_info_list (List[Dict]): List of table information dictionaries
        pdf_path (str): Original PDF file path
        output_dir (str): Directory to save CSV files
        
    Returns:
        List[str]: List of saved file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    pdf_name = Path(pdf_path).stem
    
    for i, table_info in enumerate(table_info_list):
        df = table_info['dataframe']
        page_num = table_info['page_number']
        method = table_info['extraction_method']
        
        filename = f"{pdf_name}_page{page_num}_table{i+1}_{method}.csv"
        file_path = os.path.join(output_dir, filename)
        df.to_csv(file_path, index=False)
        saved_files.append(file_path)
    
    return saved_files


def pdf_page_to_image(pdf_path: str, page_number: int, dpi: int = 150) -> Image.Image:
    """
    Convert a PDF page to a PIL Image.
    
    Args:
        pdf_path (str): Path to the PDF file
        page_number (int): Page number (1-indexed)
        dpi (int): Resolution for the image
        
    Returns:
        PIL.Image: The page as an image
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1)  # PyMuPDF uses 0-indexed pages
    
    # Convert to image
    mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 is the default DPI
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    
    # Convert to PIL Image
    image = Image.open(io.BytesIO(img_data))
    doc.close()
    
    return image


def encode_image_to_base64(image: Image.Image) -> str:
    """
    Encode PIL Image to base64 string.
    
    Args:
        image (PIL.Image): The image to encode
        
    Returns:
        str: Base64 encoded image
    """
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def summarize_page_with_multimodal_model(image: Image.Image, 
                                       openai_api_key: str = None,
                                       model_name: str = "gpt-4-vision-preview") -> str:
    """
    Summarize a PDF page using a multimodal model.
    
    Args:
        image (PIL.Image): The page image
        openai_api_key (str): OpenAI API key (if using OpenAI models)
        model_name (str): Name of the multimodal model to use
        
    Returns:
        str: Summary of the page content
    """
    try:
        # For OpenAI GPT-4 Vision
        if openai_api_key and "gpt-4" in model_name.lower():
            return _summarize_with_openai_vision(image, openai_api_key, model_name)
        
        # For free alternatives, you can add other multimodal models here
        # Example: Using a local model or free API
        else:
            return _summarize_with_free_model(image)
    
    except Exception as e:
        print(f"Error in multimodal summarization: {e}")
        return f"Error generating summary: {str(e)}"


def _summarize_with_openai_vision(image: Image.Image, api_key: str, model_name: str) -> str:
    """Summarize using OpenAI's GPT-4 Vision."""
    client = OpenAI(api_key=api_key)
    
    # Encode image
    base64_image = encode_image_to_base64(image)
    
    # Create the prompt
    prompt = """Analyze this PDF page image and provide a detailed summary focusing on:
1. Tables and data present (structure, content, key metrics)
2. Charts, graphs, or visualizations
3. Text content and key information
4. Any financial, technical, or business insights
5. Context about what this page represents

Provide a comprehensive summary that would help someone understand what data or information is available on this page."""
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=500
    )
    
    return response.choices[0].message.content


def _summarize_with_free_model(image: Image.Image) -> str:
    """
    Summarize using a free multimodal model.
    
    You can integrate free alternatives here such as:
    - Hugging Face Transformers (BLIP, LLaVA)
    - Local multimodal models
    - Other free APIs
    
    For now, this is a placeholder that returns a basic description.
    """
    # Placeholder - replace with actual free multimodal model
    # Example using Hugging Face BLIP:
    
    try:
        # This is a simplified example - you would need to install transformers
        # and implement the actual model inference
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_length=100)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        return f"Page contains: {caption}. This appears to be a document page with structured data and text content."
    
    except ImportError:
        # Fallback if transformers is not installed
        return "Page contains structured data including tables and text content. Detailed analysis requires multimodal model setup."


def create_page_summaries(pdf_path: str, page_numbers: List[int], 
                         openai_api_key: str = None) -> Dict[int, str]:
    """
    Create summaries for each page using a multimodal model.
    
    Args:
        pdf_path (str): Path to the PDF file
        page_numbers (List[int]): List of page numbers to process
        openai_api_key (str): OpenAI API key (optional)
        
    Returns:
        Dict[int, str]: Dictionary mapping page numbers to summaries
    """
    page_summaries = {}
    
    print(f"Creating summaries for {len(page_numbers)} pages...")
    
    for page_num in page_numbers:
        try:
            print(f"  Processing page {page_num}...")
            
            # Convert PDF page to image
            image = pdf_page_to_image(pdf_path, page_num)
            
            # Generate summary using multimodal model
            summary = summarize_page_with_multimodal_model(
                image, 
                openai_api_key=openai_api_key
            )
            
            page_summaries[page_num] = summary
            print(f"    ✓ Generated summary for page {page_num}")
            
            # Add delay to avoid API rate limits
            if openai_api_key:
                time.sleep(1)
                
        except Exception as e:
            print(f"    ✗ Error processing page {page_num}: {e}")
            page_summaries[page_num] = f"Error generating summary for page {page_num}: {str(e)}"
    
    return page_summaries


def create_simplified_metadata(table_info_list: List[Dict], pdf_path: str, 
                              saved_files: List[str], 
                              page_summaries: Dict[int, str]) -> List[Dict]:
    """
    Create simplified metadata containing only summary and CSV file reference.
    
    Args:
        table_info_list (List[Dict]): List of table information dictionaries
        pdf_path (str): Original PDF file path
        saved_files (List[str]): List of saved CSV file paths
        page_summaries (Dict[int, str]): Dictionary of page summaries
        
    Returns:
        List[Dict]: List of dictionaries containing minimal metadata
    """
    metadata_list = []
    
    for i, table_info in enumerate(table_info_list):
        page_num = table_info['page_number']
        method = table_info['extraction_method']
        
        # Get the summary for this page
        page_summary = page_summaries.get(page_num, "No summary available")
        
        metadata = {
            'table_id': f"{Path(pdf_path).stem}_page{page_num}_table{i+1}_{method}",
            'csv_file_path': saved_files[i] if i < len(saved_files) else None,
            'page_summary': page_summary,
            'page_number': page_num,
            'source_file': Path(pdf_path).name,
            'created_at': pd.Timestamp.now().isoformat()
        }
        metadata_list.append(metadata)
    
    return metadata_list


def create_summary_embeddings(metadata_list: List[Dict], model_name: str = 'all-MiniLM-L6-v2') -> List[Dict]:
    """
    Create embeddings from page summaries.
    
    Args:
        metadata_list (List[Dict]): List of metadata dictionaries
        model_name (str): Name of the sentence transformer model
        
    Returns:
        List[Dict]: Metadata with embeddings added
    """
    model = SentenceTransformer(model_name)
    
    for metadata in metadata_list:
        # Create embedding from the page summary
        summary_text = metadata['page_summary']
        
        # Create embedding
        embedding = model.encode(summary_text)
        
        # Add embedding to metadata
        metadata['embedding'] = embedding.tolist()
        metadata['embedding_model'] = model_name
    
    return metadata_list


def setup_weaviate_client() -> weaviate.WeaviateClient:
    """Setup Weaviate client connection."""
    weaviate_url = "https://6rbkz8vnqryn3yw5qtqkpg.c0.asia-southeast1.gcp.weaviate.cloud"
    weaviate_api_key = "U1U3TkVDalVKOGxZQmdicV9TT2xWeElSS0IvK2dKTlJLTk1zYXFoSDVDWmdtZDgxYWRJeVYrZHJpMUJVPV92MjAw"

    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
        )
        
        # Test the connection
        if client.is_ready():
            print("✓ Successfully connected to Weaviate Cloud")
        else:
            print("✗ Weaviate client not ready")
            
        return client
        
    except Exception as e:
        print(f"✗ Error connecting to Weaviate: {e}")
        raise


def create_simplified_weaviate_schema(client: weaviate.WeaviateClient, class_name: str = "PDFTableSummary") -> None:
    """Create simplified Weaviate schema for storing page summaries and CSV references."""
    
    # Delete collection if it exists
    if client.collections.exists(class_name):
        client.collections.delete(class_name)
    
    # Create collection with simplified schema
    client.collections.create(
        name=class_name,
        description="PDF page summaries with CSV table references",
        properties=[
            Property(name="table_id", data_type=DataType.TEXT, description="Unique identifier for the table"),
            Property(name="csv_file_path", data_type=DataType.TEXT, description="Path to saved CSV file"),
            Property(name="page_summary", data_type=DataType.TEXT, description="Multimodal model summary of the page"),
            Property(name="page_number", data_type=DataType.INT, description="Page number in the PDF"),
            Property(name="source_file", data_type=DataType.TEXT, description="Source PDF file name"),
            Property(name="embedding_model", data_type=DataType.TEXT, description="Model used for creating embeddings"),
            Property(name="created_at", data_type=DataType.TEXT, description="Timestamp when the data was created")
        ],
        vectorizer_config=Configure.Vectorizer.none()
    )


def store_summaries_in_weaviate(client: weaviate.WeaviateClient, metadata_with_embeddings: List[Dict], 
                               class_name: str = "PDFTableSummary") -> List[str]:
    """Store simplified metadata with summaries and embeddings in Weaviate."""
    
    collection = client.collections.get(class_name)
    object_ids = []
    
    print(f"Storing {len(metadata_with_embeddings)} summary objects...")
    
    for i, metadata in enumerate(metadata_with_embeddings):
        try:
            # Prepare data object
            data_object = {
                "table_id": metadata['table_id'],
                "csv_file_path": metadata['csv_file_path'],
                "page_summary": metadata['page_summary'],
                "page_number": metadata['page_number'],
                "source_file": metadata['source_file'],
                "embedding_model": metadata['embedding_model'],
                "created_at": metadata['created_at']
            }
            
            # Insert object with vector
            object_id = collection.data.insert(
                properties=data_object,
                vector=metadata['embedding']
            )
            
            object_ids.append(str(object_id))
            print(f"  ✓ Stored object {i+1}/{len(metadata_with_embeddings)}: {metadata['table_id']}")
            
        except Exception as e:
            print(f"  ✗ Error storing object {i+1}: {e}")
            continue
    
    return object_ids


def search_tables_by_summary(client: weaviate.WeaviateClient, query_text: str, 
                           model_name: str = 'all-MiniLM-L6-v2', 
                           class_name: str = "PDFTableSummary", 
                           limit: int = 5) -> List[Dict]:
    """Search for tables using semantic similarity on page summaries."""
    
    try:
        # Create embedding for the query
        model = SentenceTransformer(model_name)
        query_embedding = model.encode(query_text).tolist()
        
        # Get collection and perform vector search
        collection = client.collections.get(class_name)
        
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=limit,
            return_metadata=MetadataQuery(certainty=True, distance=True)
        )
        
        # Format results
        results = []
        for obj in response.objects:
            metadata_dict = {}
            if hasattr(obj.metadata, 'certainty'):
                metadata_dict['certainty'] = obj.metadata.certainty
            if hasattr(obj.metadata, 'distance'):
                metadata_dict['distance'] = obj.metadata.distance
            
            result = {
                "id": str(obj.uuid),
                "properties": obj.properties,
                "metadata": metadata_dict
            }
            results.append(result)
        
        return results
    
    except Exception as e:
        print(f"Error in search: {e}")
        return []


def load_csv_from_search_result(csv_file_path: str) -> pd.DataFrame:
    """
    Load the CSV file referenced in search results.
    
    Args:
        csv_file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: The loaded CSV data
    """
    try:
        if os.path.exists(csv_file_path):
            return pd.read_csv(csv_file_path)
        else:
            print(f"CSV file not found: {csv_file_path}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()


def process_pdf_with_multimodal_pipeline(pdf_path: str, 
                                        output_dir: str = "table_data", 
                                        embedding_model: str = 'all-MiniLM-L6-v2',
                                        openai_api_key: str = None) -> List[str]:
    """
    Complete pipeline with multimodal page summarization.
    
    Args:
        pdf_path (str): Path to PDF file
        output_dir (str): Directory to save CSV files
        embedding_model (str): Embedding model name
        openai_api_key (str): OpenAI API key for GPT-4 Vision (optional)
        
    Returns:
        List[str]: List of created object IDs in Weaviate
    """
    client = None
    
    try:
        print(f"Processing PDF with multimodal pipeline: {pdf_path}")
        
        # Step 1: Identify potential table pages
        potential_pages = identify_potential_table_pages(pdf_path)
        
        if not potential_pages:
            print("No potential table pages found")
            return []
        
        # Step 2: Extract tables from pages
        all_tables = extract_tables_from_pages(pdf_path, potential_pages)
        
        if not all_tables:
            print("No tables extracted from the PDF")
            return []
        
        print(f"Successfully extracted {len(all_tables)} tables total")
        
        # Step 3: Save DataFrames to CSV files
        saved_files = save_dataframes_to_files(all_tables, pdf_path, output_dir)
        print(f"Saved {len(saved_files)} CSV files to {output_dir}")
        
        # Step 4: Create page summaries using multimodal model
        page_summaries = create_page_summaries(pdf_path, potential_pages, openai_api_key)
        print(f"Generated summaries for {len(page_summaries)} pages")
        
        # Step 5: Create simplified metadata with summaries
        metadata_list = create_simplified_metadata(all_tables, pdf_path, saved_files, page_summaries)
        
        # Step 6: Create embeddings from summaries
        metadata_with_embeddings = create_summary_embeddings(metadata_list, embedding_model)
        
        # Step 7: Setup Weaviate and store metadata
        client = setup_weaviate_client()
        create_simplified_weaviate_schema(client)
        object_ids = store_summaries_in_weaviate(client, metadata_with_embeddings)
        
        print(f"Stored metadata for {len(object_ids)} tables in Weaviate")
        return object_ids
    
    except Exception as e:
        print(f"Error in pipeline: {e}")
        raise
    
    finally:
        if client:
            client.close()


def search_and_retrieve_tables(query_text: str, 
                             embedding_model: str = 'all-MiniLM-L6-v2',
                             limit: int = 3) -> List[Dict]:
    """
    Search for relevant tables and return both summaries and CSV data.
    
    Args:
        query_text (str): Query text to search for
        embedding_model (str): Embedding model name
        limit (int): Maximum number of results
        
    Returns:
        List[Dict]: List of search results with CSV data loaded
    """
    client = None
    
    try:
        # Setup Weaviate client
        client = setup_weaviate_client()
        
        # Search using summaries
        search_results = search_tables_by_summary(client, query_text, embedding_model, limit=limit)
        
        # Load CSV data for each result
        enhanced_results = []
        
        for result in search_results:
            props = result['properties']
            csv_path = props['csv_file_path']
            
            # Load the CSV data
            csv_data = load_csv_from_search_result(csv_path)
            
            enhanced_result = {
                'table_id': props['table_id'],
                'page_summary': props['page_summary'],
                'csv_file_path': csv_path,
                'page_number': props['page_number'],
                'source_file': props['source_file'],
                'csv_data': csv_data,
                'search_certainty': result['metadata'].get('certainty', 0),
                'search_distance': result['metadata'].get('distance', 1)
            }
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    except Exception as e:
        print(f"Error in search and retrieval: {e}")
        return []
    
    finally:
        if client:
            client.close()


# Example usage
if __name__ == "__main__":
    pdf_file_path = "pdf_input.pdf"
    openai_api_key = None
    
    try:
        # Process PDF with multimodal pipeline
        object_ids = process_pdf_with_multimodal_pipeline(
            pdf_file_path, 
            openai_api_key=openai_api_key
        )
        print(f"Created {len(object_ids)} objects in Weaviate")
        
        # Example search and retrieval
        query = "financial revenue data quarterly results"
        results = search_and_retrieve_tables(query, limit=3)
        
        print(f"\nSearch results for: '{query}'")
        print("=" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Table ID: {result['table_id']}")
            print(f"Source: {result['source_file']} (Page {result['page_number']})")
            print(f"Certainty: {result['search_certainty']:.3f}")
            print(f"CSV Path: {result['csv_file_path']}")
            print(f"Summary: {result['page_summary'][:200]}...")
            
            # Show CSV data preview
            if not result['csv_data'].empty:
                print(f"CSV Shape: {result['csv_data'].shape}")
                print("CSV Preview:")
                print(result['csv_data'].head(3).to_string())
            else:
                print("No CSV data available")
            
            print("-" * 30)
        
    except Exception as e:
        print(f"Error: {e}")