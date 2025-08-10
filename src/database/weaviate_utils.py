import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.init import Auth

def setup_weaviate_client(url, api_key):
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(api_key),
    )
    return client

def create_schema(client, class_name="PDFTableSummary"):
    if client.collections.exists(class_name):
        client.collections.delete(class_name)
    client.collections.create(
        name=class_name,
        description="PDF page summaries with CSV table references",
        properties=[
            Property(name="table_id", data_type=DataType.TEXT),
            Property(name="csv_file_path", data_type=DataType.TEXT),
            Property(name="page_summary", data_type=DataType.TEXT),
            Property(name="page_number", data_type=DataType.INT),
            Property(name="source_file", data_type=DataType.TEXT),
            Property(name="embedding_model", data_type=DataType.TEXT),
            Property(name="created_at", data_type=DataType.TEXT)
        ],
        vectorizer_config=Configure.Vectorizer.none()
    )

def store_in_weaviate(client, data, class_name="PDFTableSummary"):
    col = client.collections.get(class_name)
    for item in data:
        col.data.insert(properties={k: v for k, v in item.items() if k != "embedding"},
                        vector=item["embedding"])

def get_weaviate_client(url: str, api_key: str):
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(api_key),
    )

def search_tables(client, query_vector, top_k: int = 3):
    collection = client.collections.get("PDFTableSummary")
    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
        return_metadata=["certainty"]
    )
    results = []
    for obj in response.objects:
        results.append({
            "table_id": obj.properties.get("table_id"),
            "csv_file_path": obj.properties.get("csv_file_path"),
            "page_summary": obj.properties.get("page_summary"),
            "page_number": obj.properties.get("page_number"),
            "source_file": obj.properties.get("source_file"),
            "certainty": obj.metadata.certainty if obj.metadata else 0.0
        })
    return results