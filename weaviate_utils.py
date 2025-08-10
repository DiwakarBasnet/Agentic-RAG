# weaviate_utils.py
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
from weaviate.classes.config import Configure, Property, DataType
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from config import WEAVIATE_URL, WEAVIATE_API_KEY, WEAVIATE_CLASS

def search_similar_tables(
    client,
    query_text: str,
    model_name: str = "all-MiniLM-L6-v2",
    class_name: str = WEAVIATE_CLASS,
    limit: int = 5
) -> List[Dict]:
    """
    Search for similar tables using semantic similarity.

    Args:
        client: Weaviate client instance.
        query_text (str): Text to search for.
        model_name (str): SentenceTransformer model name.
        class_name (str): Weaviate collection name.
        limit (int): Number of results to return.

    Returns:
        List[Dict]: Matching table metadata and similarity scores.
    """
    try:
        model = SentenceTransformer(model_name)
        query_embedding = model.encode(query_text).tolist()

        collection = client.collections.get(class_name)
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=limit,
            return_metadata=MetadataQuery(certainty=True, distance=True)
        )
        results = []
        for obj in response.objects:
            metadata_dict = {
                "certainty": getattr(obj.metadata, "certainty", None),
                "distance": getattr(obj.metadata, "distance", None)
            }
            results.append({
                "id": str(obj.uuid),
                "properties": obj.properties,
                "metadata": metadata_dict
            })

        return results
    except Exception as e:
        return []

def setup_weaviate_client():
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    )

def create_weaviate_schema(client):
    if client.collections.exists(WEAVIATE_CLASS):
        client.collections.delete(WEAVIATE_CLASS)
    client.collections.create(
        name=WEAVIATE_CLASS,
        description="PDF table metadata with embeddings",
        properties=[
            Property(name="table_id", data_type=DataType.TEXT),
            Property(name="source_file", data_type=DataType.TEXT),
            Property(name="page_number", data_type=DataType.INT),
            Property(name="table_index", data_type=DataType.INT),
            Property(name="extraction_method", data_type=DataType.TEXT),
            Property(name="csv_file_path", data_type=DataType.TEXT),
            Property(name="shape", data_type=DataType.TEXT),
            Property(name="columns", data_type=DataType.TEXT_ARRAY),
            Property(name="row_count", data_type=DataType.INT),
            Property(name="column_count", data_type=DataType.INT),
            Property(name="sample_text", data_type=DataType.TEXT),
            Property(name="embedding_model", data_type=DataType.TEXT),
            Property(name="created_at", data_type=DataType.TEXT),
        ],
        vectorizer_config=Configure.Vectorizer.none(),
    )

def store_metadata_in_weaviate(client, metadata_with_embeddings: List[Dict]) -> List[str]:
    collection = client.collections.get(WEAVIATE_CLASS)
    ids = []
    for metadata in metadata_with_embeddings:
        obj = {k: v for k, v in metadata.items() if k not in ["embedding", "dataframe"]}
        obj["shape"] = f"{metadata['shape'][0]}x{metadata['shape'][1]}"
        oid = collection.data.insert(properties=obj, vector=metadata["embedding"])
        ids.append(str(oid))
    return ids

def create_weaviate_text_schema(client):
    """Create schema for storing plain PDF page text embeddings."""
    if "PDFTextMetadata" in [c.name for c in client.collections.list()]:
        return

    client.collections.create(
        name="PDFTextMetadata",
        vectorizer_config=None,
        properties=[
            {"name": "text_id", "dataType": ["string"]},
            {"name": "source_file", "dataType": ["string"]},
            {"name": "page_number", "dataType": ["int"]},
            {"name": "page_title", "dataType": ["string"]},
            {"name": "sample_text", "dataType": ["string"]}
        ]
    )