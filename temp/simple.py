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


def create_table_metadata(table_info_list: List[Dict], pdf_path: str, 
                         saved_files: List[str]) -> List[Dict]:
    """
    Create metadata for tables without storing the actual DataFrame data.
    
    Args:
        table_info_list (List[Dict]): List of table information dictionaries
        pdf_path (str): Original PDF file path
        saved_files (List[str]): List of saved CSV file paths
        
    Returns:
        List[Dict]: List of dictionaries containing table metadata
    """
    metadata_list = []
    
    for i, table_info in enumerate(table_info_list):
        df = table_info['dataframe']
        page_num = table_info['page_number']
        method = table_info['extraction_method']
        
        # Create sample text representation (first few rows)
        sample_text_parts = []
        
        if not df.empty:
            # Convert column names to strings (fix for Weaviate text array requirement)
            column_names = [str(col) for col in df.columns]
            
            # Add column headers
            headers = " | ".join(column_names)
            sample_text_parts.append(f"Headers: {headers}")
            
            # Add first 3 rows as sample
            sample_rows = min(3, len(df))
            for idx in range(sample_rows):
                row_text = " | ".join([str(val) for val in df.iloc[idx].values])
                sample_text_parts.append(f"Row {idx+1}: {row_text}")
        
        sample_text = "\n".join(sample_text_parts)
        
        metadata = {
            'table_id': f"{Path(pdf_path).stem}_page{page_num}_table{i+1}_{method}",
            'source_file': pdf_path,
            'page_number': page_num,
            'table_index': i,
            'extraction_method': method,
            'csv_file_path': saved_files[i] if i < len(saved_files) else None,
            'shape': df.shape,
            'columns': [str(col) for col in df.columns],  # Ensure all columns are strings
            'row_count': len(df),
            'column_count': len(df.columns),
            'sample_text': sample_text,
            'parsing_report': table_info.get('parsing_report'),
            'created_at': pd.Timestamp.now().isoformat()
        }
        metadata_list.append(metadata)
    
    return metadata_list


def create_embeddings(metadata_list: List[Dict], model_name: str = 'all-MiniLM-L6-v2') -> List[Dict]:
    """
    Create semantic embeddings for table metadata.
    
    Args:
        metadata_list (List[Dict]): List of table metadata
        model_name (str): Name of the sentence transformer model
        
    Returns:
        List[Dict]: Metadata with embeddings added
    """
    model = SentenceTransformer(model_name)
    
    for metadata in metadata_list:
        # Create embedding text from metadata
        embedding_text = f"Table from {metadata['source_file']} page {metadata['page_number']}. {metadata['sample_text']}"
        
        # Create embedding
        embedding = model.encode(embedding_text)
        
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


def create_weaviate_schema(client: weaviate.WeaviateClient, class_name: str = "PDFTableMetadata") -> None:
    """Create Weaviate schema for storing PDF table metadata only."""
    
    # Delete collection if it exists
    if client.collections.exists(class_name):
        client.collections.delete(class_name)
    
    # Create collection with schema
    client.collections.create(
        name=class_name,
        description="PDF table metadata with embeddings",
        properties=[
            Property(name="table_id", data_type=DataType.TEXT, description="Unique identifier for the table"),
            Property(name="source_file", data_type=DataType.TEXT, description="Source PDF file path"),
            Property(name="page_number", data_type=DataType.INT, description="Page number in the PDF"),
            Property(name="table_index", data_type=DataType.INT, description="Index of table on the page"),
            Property(name="extraction_method", data_type=DataType.TEXT, description="Method used to extract table (lattice/stream)"),
            Property(name="csv_file_path", data_type=DataType.TEXT, description="Path to saved CSV file"),
            Property(name="shape", data_type=DataType.TEXT, description="Shape of the DataFrame (rows, columns)"),
            Property(name="columns", data_type=DataType.TEXT_ARRAY, description="Column names of the DataFrame"),
            Property(name="row_count", data_type=DataType.INT, description="Number of rows in the table"),
            Property(name="column_count", data_type=DataType.INT, description="Number of columns in the table"),
            Property(name="sample_text", data_type=DataType.TEXT, description="Sample text representation of the table"),
            Property(name="embedding_model", data_type=DataType.TEXT, description="Model used for creating embeddings"),
            Property(name="created_at", data_type=DataType.TEXT, description="Timestamp when the data was created")
        ],
        vectorizer_config=Configure.Vectorizer.none()
    )


def store_metadata_in_weaviate(client: weaviate.WeaviateClient, metadata_with_embeddings: List[Dict], 
                              class_name: str = "PDFTableMetadata") -> List[str]:
    """Store table metadata and embeddings in Weaviate."""
    
    collection = client.collections.get(class_name)
    object_ids = []
    
    print(f"Attempting to store {len(metadata_with_embeddings)} objects...")
    
    for i, metadata in enumerate(metadata_with_embeddings):
        try:
            # Prepare metadata object (excluding the embedding and dataframe)
            data_object = {
                "table_id": metadata['table_id'],
                "source_file": metadata['source_file'],
                "page_number": metadata['page_number'],
                "table_index": metadata['table_index'],
                "extraction_method": metadata['extraction_method'],
                "csv_file_path": metadata['csv_file_path'],
                "shape": f"{metadata['shape'][0]}x{metadata['shape'][1]}",
                "columns": metadata['columns'],
                "row_count": metadata['row_count'],
                "column_count": metadata['column_count'],
                "sample_text": metadata['sample_text'],
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
            print(f"    Table ID: {metadata.get('table_id', 'Unknown')}")
            continue
    
    # Verify objects were actually stored
    try:
        total_objects = collection.aggregate.over_all(total_count=True)
        print(f"Collection now contains {total_objects.total_count} total objects")
    except Exception as e:
        print(f"Could not verify object count: {e}")
    
    return object_ids


def debug_weaviate_connection(client: weaviate.WeaviateClient) -> None:
    """Debug function to check Weaviate connection and collections."""
    try:
        print("\n=== WEAVIATE DEBUG INFO ===")
        
        # Check if client is ready
        print(f"Client ready: {client.is_ready()}")
        
        # List all collections
        collections = client.collections.list_all()
        print(f"Available collections: {[c.name for c in collections]}")
        
        # Check if our collection exists and get details
        class_name = "PDFTableMetadata"
        if client.collections.exists(class_name):
            collection = client.collections.get(class_name)
            
            # Get collection info
            config = collection.config.get()
            print(f"Collection '{class_name}' exists")
            print(f"Collection properties: {[p.name for p in config.properties]}")
            
            # Get object count
            try:
                result = collection.aggregate.over_all(total_count=True)
                print(f"Object count: {result.total_count}")
                
                # Try to fetch first few objects if any exist
                if result.total_count > 0:
                    objects = collection.query.fetch_objects(limit=3)
                    print(f"Sample objects: {len(objects.objects)} found")
                    for obj in objects.objects[:2]:  # Show first 2
                        print(f"  - ID: {obj.uuid}")
                        print(f"    table_id: {obj.properties.get('table_id', 'N/A')}")
            except Exception as e:
                print(f"Error getting object count: {e}")
                
        else:
            print(f"Collection '{class_name}' does not exist")
            
        print("=== END DEBUG INFO ===\n")
        
    except Exception as e:
        print(f"Debug error: {e}")


def test_simple_insert(client: weaviate.WeaviateClient) -> bool:
    """Test inserting a simple object to verify Weaviate is working."""
    try:
        class_name = "PDFTableMetadata"
        collection = client.collections.get(class_name)
        
        # Create a simple test object
        test_object = {
            "table_id": "test_table_123",
            "source_file": "test.pdf",
            "page_number": 1,
            "table_index": 0,
            "extraction_method": "test",
            "csv_file_path": "test.csv",
            "shape": "2x3",
            "columns": ["col1", "col2", "col3"],
            "row_count": 2,
            "column_count": 3,
            "sample_text": "Test table data",
            "embedding_model": "test-model",
            "created_at": pd.Timestamp.now().isoformat()
        }
        
        # Create a dummy embedding vector (384 dimensions for all-MiniLM-L6-v2)
        test_embedding = [0.0] * 384
        
        # Insert test object
        object_id = collection.data.insert(
            properties=test_object,
            vector=test_embedding
        )
        
        print(f"✓ Test insert successful. Object ID: {object_id}")
        return True
        
    except Exception as e:
        print(f"✗ Test insert failed: {e}")
        return False
def search_similar_tables(client: weaviate.WeaviateClient, query_text: str, 
                         model_name: str = 'all-MiniLM-L6-v2', 
                         class_name: str = "PDFTableMetadata", 
                         limit: int = 5) -> List[Dict]:
    """Search for similar tables using semantic similarity."""
    
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
        
        # Format results with proper metadata handling
        results = []
        for obj in response.objects:
            # Handle metadata properly - it's an object, not a dict
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


def process_pdf_pipeline(pdf_path: str, output_dir: str = "table_data", 
                        embedding_model: str = 'all-MiniLM-L6-v2') -> List[str]:
    """
    Complete pipeline to process PDF using unstructured elements detection and store metadata in Weaviate.
    
    Args:
        pdf_path (str): Path to PDF file
        output_dir (str): Directory to save CSV files
        embedding_model (str): Embedding model name
        
    Returns:
        List[str]: List of created object IDs in Weaviate
    """
    client = None
    
    try:
        print(f"Processing PDF: {pdf_path}")
        
        # Step 1: Identify potential table pages using unstructured
        potential_pages = identify_potential_table_pages(pdf_path)
        
        if not potential_pages:
            print("No potential table pages found")
            return []
        
        # Step 2: Extract tables from potential pages using both Camelot flavors
        all_tables = extract_tables_from_pages(pdf_path, potential_pages)
        
        if not all_tables:
            print("No tables extracted from the PDF")
            return []
        
        print(f"Successfully extracted {len(all_tables)} tables total")
        
        # Step 3: Save DataFrames to CSV files
        saved_files = save_dataframes_to_files(all_tables, pdf_path, output_dir)
        print(f"Saved {len(saved_files)} CSV files to {output_dir}")
        
        # Step 4: Create metadata
        metadata_list = create_table_metadata(all_tables, pdf_path, saved_files)
        
        # Step 5: Create embeddings
        metadata_with_embeddings = create_embeddings(metadata_list, embedding_model)
        
        # Step 6: Setup Weaviate and store metadata
        client = setup_weaviate_client()
        
        # Debug the connection
        debug_weaviate_connection(client)
        
        create_weaviate_schema(client)
        
        # Test simple insert first
        if test_simple_insert(client):
            print("Basic insertion test passed, proceeding with actual data...")
        else:
            print("Basic insertion test failed, there may be a connection issue")
            return []
        
        object_ids = store_metadata_in_weaviate(client, metadata_with_embeddings)
        print(f"Stored metadata for {len(object_ids)} tables in Weaviate")
        
        # Final verification
        debug_weaviate_connection(client)
        
        return object_ids
    
    except Exception as e:
        print(f"Error in pipeline: {e}")
        raise
    
    finally:
        if client:
            client.close()


# Example usage
if __name__ == "__main__":
    pdf_file_path = "pdf_input.pdf"
    client = None
    
    try:
        # Process PDF
        object_ids = process_pdf_pipeline(pdf_file_path)
        print(f"Created {len(object_ids)} objects in Weaviate")
        
        # Debug connection and verify data
        client = setup_weaviate_client()
        debug_weaviate_connection(client)
        
        # Example search
        results = search_similar_tables(client, "financial data revenue")
        
        print("\nSearch results:")
        for result in results:
            props = result['properties']
            print(f"Table: {props['table_id']}")
            print(f"Page: {props['page_number']}, Method: {props['extraction_method']}")
            print(f"CSV: {props['csv_file_path']}")
            print(f"Shape: {props['shape']}, Certainty: {result['metadata']['certainty']:.3f}")
            print("---")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        if client:
            client.close()