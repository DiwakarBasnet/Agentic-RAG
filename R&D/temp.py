import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from datetime import datetime

# Core dependencies
import weaviate
from weaviate.classes.init import Auth
import pandas as pd
import numpy as np

# LangChain imports
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Weaviate
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Hugging Face
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    pipeline,
    BitsAndBytesConfig
)
import torch

# Document processing
import fitz  # PyMuPDF
import re
from urllib.parse import urlparse
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SECPDFProcessor:
    """Handles SEC PDF document processing and cleaning"""
    
    def __init__(self):
        self.sec_patterns = {
            'form_type': r'FORM\s+(10-K|10-Q|8-K|DEF 14A|S-1)',
            'company_name': r'COMPANY CONFORMED NAME:\s*(.+)',
            'cik': r'CENTRAL INDEX KEY:\s*(\d+)',
            'filing_date': r'FILED AS OF DATE:\s*(\d{8})',
            'period_end': r'CONFORMED PERIOD OF REPORT:\s*(\d{8})'
        }
    
    def extract_metadata(self, text: str) -> Dict[str, str]:
        """Extract SEC-specific metadata from document text"""
        metadata = {}
        for key, pattern in self.sec_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata[key] = match.group(1).strip()
        
        return metadata
    
    def clean_sec_text(self, text: str) -> str:
        """Clean and normalize SEC document text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'Table of Contents.*?(?=\n\n|\n[A-Z])', '', text, flags=re.DOTALL)
        text = re.sub(r'Page \d+ of \d+', '', text)
        
        # Clean up common SEC formatting issues
        text = re.sub(r'\$\s+(\d)', r'$\1', text)
        text = re.sub(r'(\d)\s+%', r'\1%', text)
        
        # Remove excessive line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()


class WeaviateVectorStore:
    """Manages Weaviate vector database operations"""
    
    def __init__(self, 
                 url: str = "https://6rbkz8vnqryn3yw5qtqkpg.c0.asia-southeast1.gcp.weaviate.cloud",
                 api_key: Optional[str] = "ZE1JQWhwU1MzMEZLZ2NoMF9vMmRLTERUejFaalRRRk5wZEhyM0FDZXJGeldUMUhNQVVhWXl2bVhXTzlVPV92MjAw",
                 class_name: str = "SECDocument"):
        self.url = url
        self.api_key = api_key
        self.class_name = class_name
        self.client = None
        self.collection = None
        
    def connect(self):
        """Connect to Weaviate instance"""
        try:
            if self.api_key:
                auth_config = Auth.api_key(self.api_key)
                self.client = weaviate.connect_to_wcs(
                    cluster_url=self.url,
                    auth_credentials=auth_config
                )
            else:
                self.client = weaviate.connect_to_local(
                    host=self.url.replace('http://', '').replace('https://', '').split(':')[0],
                    port=int(self.url.split(':')[-1]) if ':' in self.url else 8080
                )
            
            logger.info("Successfully connected to Weaviate")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            return False
    
    def create_schema(self):
        """Create Weaviate schema for SEC documents"""
        try:
            # Check if collection exists
            if self.client.collections.exists(self.class_name):
                logger.info(f"Collection {self.class_name} already exists")
                self.collection = self.client.collections.get(self.class_name)
                return True
            
            # Create new collection
            self.collection = self.client.collections.create(
                name=self.class_name,
                properties=[
                    weaviate.classes.config.Property(
                        name="content",
                        data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    weaviate.classes.config.Property(
                        name="source",
                        data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    weaviate.classes.config.Property(
                        name="form_type",
                        data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    weaviate.classes.config.Property(
                        name="company_name",
                        data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    weaviate.classes.config.Property(
                        name="cik",
                        data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    weaviate.classes.config.Property(
                        name="filing_date",
                        data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    weaviate.classes.config.Property(
                        name="chunk_id",
                        data_type=weaviate.classes.config.DataType.INT
                    ),
                    weaviate.classes.config.Property(
                        name="created_at",
                        data_type=weaviate.classes.config.DataType.DATE
                    )
                ],
                # Configure vectorization
                vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
            )
            
            logger.info(f"Created collection: {self.class_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            return False
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """Add documents with embeddings to Weaviate"""
        try:
            objects = []
            
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                obj = {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "form_type": doc.metadata.get("form_type", ""),
                    "company_name": doc.metadata.get("company_name", ""),
                    "cik": doc.metadata.get("cik", ""),
                    "filing_date": doc.metadata.get("filing_date", ""),
                    "chunk_id": i,
                    "created_at": datetime.now().isoformat()
                }
                
                objects.append(weaviate.classes.data.DataObject(
                    properties=obj,
                    vector=embedding
                ))
            
            # Batch insert
            response = self.collection.data.insert_many(objects)
            
            if response.has_errors:
                logger.error(f"Errors during batch insert: {response.errors}")
                return False
            
            logger.info(f"Successfully inserted {len(objects)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

class HuggingFaceEmbeddingModel:
    """Wrapper for Hugging Face embedding models."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embeddings = None
        
    def initialize(self):
        """Initialize the embedding model"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Initialized embedding model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            return False
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        return self.embeddings.embed_query(text)

class HuggingFaceLLM:
    """Wrapper for Hugging Face language models"""
    
    def __init__(self, 
                 model_name: str = "openai/gpt-oss-20b:fireworks-ai",
                 max_new_tokens: int = 512,
                 temperature: float = 0.1):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.pipeline = None
        self.llm = None
        
    def initialize(self):
        """Initialize the language model"""
        try:
            # Configure quantization for larger models
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                device_map="auto",
                model_kwargs={
                    "quantization_config": quantization_config,
                    "torch_dtype": torch.float16,
                }
            )
            
            # Wrap in LangChain
            self.llm = HuggingFacePipeline(pipeline=self.pipeline)
            
            logger.info(f"Initialized LLM: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            return False


class SECRAGSystem:
    """Main RAG system for SEC document analysis"""
    
    def __init__(self, 
                 weaviate_url: str = "https://548cae5wquiajhovivwljw.c0.asia-southeast1.gcp.weaviate.cloud",
                 weaviate_api_key: Optional[str] = "ZE1JQWhwU1MzMEZLZ2NoMF9vMmRLTERUejFaalRRRk5wZEhyM0FDZXJGeldUMUhNQVVhWXl2bVhXTzlVPV92MjAw",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_model: str = "openai/gpt-oss-20b:fireworks-ai"):
        
        self.pdf_processor = SECPDFProcessor()
        self.vector_store = WeaviateVectorStore(weaviate_url, weaviate_api_key)
        self.embedding_model = HuggingFaceEmbeddingModel(embedding_model)
        self.llm = HuggingFaceLLM(llm_model)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.qa_chain = None
        
    def initialize(self):
        """Initialize all components"""
        logger.info("Initializing SEC RAG System...")
        
        # Connect to Weaviate
        if not self.vector_store.connect():
            return False
            
        # Create schema
        if not self.vector_store.create_schema():
            return False
        
        # Initialize models
        if not self.embedding_model.initialize():
            return False
            
        if not self.llm.initialize():
            return False
        
        logger.info("SEC RAG System initialized successfully!")
        return True
    
    def process_pdf_file(self, pdf_path: str) -> List[Document]:
        """Process a single PDF file"""
        try:
            # Load PDF
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            
            # Process each page
            processed_docs = []
            
            for doc in documents:
                # Extract metadata
                metadata = self.pdf_processor.extract_metadata(doc.page_content)
                
                # Clean text
                cleaned_text = self.pdf_processor.clean_sec_text(doc.page_content)
                
                # Update metadata
                doc.metadata.update(metadata)
                doc.metadata["source"] = pdf_path
                doc.page_content = cleaned_text
                
                processed_docs.append(doc)
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(processed_docs)
            
            logger.info(f"Processed {pdf_path}: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            return []
    
    def ingest_documents(self, pdf_paths: List[str]):
        """Ingest multiple PDF documents into the vector store"""
        all_chunks = []
        
        for pdf_path in pdf_paths:
            chunks = self.process_pdf_file(pdf_path)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.error("No documents to ingest")
            return False
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = [doc.page_content for doc in all_chunks]
        embeddings = self.embedding_model.embed_documents(texts)
        
        # Add to vector store
        success = self.vector_store.add_documents(all_chunks, embeddings)
        
        if success:
            logger.info(f"Successfully ingested {len(all_chunks)} document chunks")
        
        return success
    
    def setup_qa_chain(self):
        """Setup the QA retrieval chain"""
        try:
            # Create LangChain Weaviate retriever
            vectorstore = Weaviate(
                client=self.vector_store.client,
                index_name=self.vector_store.class_name,
                text_key="content",
                embedding=self.embedding_model.embeddings,
                by_text=False,
                attributes=["source", "form_type", "company_name", "cik", "filing_date"]
            )
            
            # Create retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Define prompt template
            prompt_template = """You are an expert SEC document analyst. Use the following pieces of context from SEC filings to answer the question. If you cannot answer the question based on the context, say so clearly.

Context:
{context}

Question: {question}

Provide a detailed answer based on the SEC filing information above. Include specific details like company names, dates, financial figures, and form types when available.

Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            logger.info("QA chain setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup QA chain: {e}")
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system."""
        if not self.qa_chain:
            if not self.setup_qa_chain():
                return {"error": "Failed to setup QA chain"}
        
        try:
            result = self.qa_chain({"query": question})
            
            # Format response
            response = {
                "question": question,
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content[:500] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ]
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"error": str(e)}
    
    def search_by_company(self, company_name: str, limit: int = 10) -> List[Dict]:
        """Search documents by company name"""
        try:
            response = self.vector_store.collection.query.bm25(
                query=company_name,
                where=weaviate.classes.query.Filter.by_property("company_name").contains_any([company_name]),
                limit=limit,
                return_metadata=weaviate.classes.query.MetadataQuery(score=True)
            )
            
            results = []
            for obj in response.objects:
                results.append({
                    "content": obj.properties["content"][:200] + "...",
                    "metadata": {
                        "company_name": obj.properties.get("company_name"),
                        "form_type": obj.properties.get("form_type"),
                        "filing_date": obj.properties.get("filing_date"),
                        "source": obj.properties.get("source")
                    },
                    "score": obj.metadata.score
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Company search failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            # Get document count
            response = self.vector_store.collection.aggregate.over_all(
                total_count=True
            )
            
            doc_count = response.total_count
            
            # Get company distribution
            company_response = self.vector_store.collection.aggregate.over_all(
                group_by="company_name"
            )
            
            companies = {}
            for group in company_response.groups:
                company = group.grouped_by["value"]
                companies[company] = group.total_count
            
            return {
                "total_documents": doc_count,
                "companies": companies,
                "system_status": "operational"
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

# Example usage and testing
def main():
    """Example usage of the SEC RAG System"""
    
    # Initialize the system
    rag_system = SECRAGSystem(
        weaviate_url="https://548cae5wquiajhovivwljw.c0.asia-southeast1.gcp.weaviate.cloud",  # Update with your Weaviate URL
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="openai/gpt-oss-20b:fireworks-ai"  # You might want to use a larger model
    )
    
    # Initialize components
    if not rag_system.initialize():
        logger.error("Failed to initialize RAG system")
        return
    
    # Example: Ingest PDF files
    pdf_files = [
        # "path/to/apple_10k.pdf",
        # "path/to/microsoft_10q.pdf",
        # Add your SEC PDF files here
        "data/inputs/pdf_input.pdf"
    ]
    
    # Process and ingest documents
    if pdf_files:
        logger.info("Starting document ingestion...")
        rag_system.ingest_documents(pdf_files)
    
    # Example queries
    questions = [
        "What was Apple's revenue in the most recent quarter?",
        "What are the main risk factors mentioned by Microsoft?",
        "Which companies filed 10-K forms?",
        "What are the key business segments mentioned in the filings?"
    ]
    
    # Query the system
    for question in questions:
        logger.info(f"\nQuestion: {question}")
        result = rag_system.query(question)
        
        if "error" in result:
            logger.error(f"Error: {result['error']}")
        else:
            logger.info(f"Answer: {result['answer']}")
            logger.info(f"Sources: {len(result['source_documents'])} documents")
    
    # Get system statistics
    stats = rag_system.get_stats()
    logger.info(f"\nSystem Stats: {stats}")

if __name__ == "__main__":
    main()