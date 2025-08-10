import os
import pandas as pd
import torch
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config.config import (
    OUTPUT_DIR, EMBEDDING_MODEL, MULTI_MODAL_MODEL, 
    WEAVIATE_URL, WEAVIATE_API_KEY, LLM_MODEL
)
from src.processing.pdf_utils import identify_potential_table_pages, pdf_page_to_image
from src.processing.table_extraction import extract_tables_from_pages, save_dataframes_to_files
from src.processing.summarization import BLIPSummarizer
from src.models.metadata import create_simplified_metadata, create_summary_embeddings, RAGAnswer
from src.database.weaviate_utils import (
    setup_weaviate_client, create_schema, store_in_weaviate, search_tables
)

_PROMPT_SNIPPET = "Analyze this PDF page image and provide a detailed summary"

def _sanitize_summary(summary: str) -> str:
    if not summary:
        return summary
    if _PROMPT_SNIPPET in summary:
        idx = summary.find(_PROMPT_SNIPPET)
        cleaned = summary[:idx] + summary[idx + len(_PROMPT_SNIPPET):]
        return cleaned.strip()
    return summary.strip()

class SimpleRAG:
    def __init__(self, weaviate_url=WEAVIATE_URL, weaviate_api_key=WEAVIATE_API_KEY,
                 embedding_model=EMBEDDING_MODEL, llm_model=LLM_MODEL):
        self.encoder = SentenceTransformer(embedding_model)
        self.client = setup_weaviate_client(weaviate_url, weaviate_api_key)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model, torch_dtype="auto", device_map="auto"
        )

    def search_relevant_tables(self, query: str, top_k: int = 3) -> List[Dict]:
        query_vector = self.encoder.encode(query).tolist()
        return search_tables(self.client, query_vector, top_k)

    def load_table_data(self, csv_file_path: str, max_rows: int = 5) -> Optional[str]:
        if not os.path.exists(csv_file_path):
            return None
        df = pd.read_csv(csv_file_path)
        if len(df) > max_rows:
            df = df.head(max_rows)
            if len(df.columns) > 4:
                df = df.iloc[:, :4]
            table_str = df.to_string(index=False, max_cols=4)
            table_str += f"\n... (showing first {max_rows} rows of {len(pd.read_csv(csv_file_path))} total rows)"
        else:
            if len(df.columns) > 4:
                df = df.iloc[:, :4]
            table_str = df.to_string(index=False, max_cols=4)
        return table_str

    def generate_answer(self, query: str, context: str) -> RAGAnswer:
        prompt = f"Based on this financial data, answer the question directly.\n\nData: {context[:600]}\n\nQuestion: {query}\n\nAnswer:"
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=100, temperature=0.1, top_p=0.7,
            do_sample=False, pad_token_id=self.tokenizer.eos_token_id, repetition_penalty=1.1
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        full_output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return RAGAnswer(thinking="", answer=full_output, raw_answer=full_output)

    def answer_question(self, query: str, top_k: int = 2, include_table_data: bool = True, show_thinking: bool = False) -> Dict:
        relevant_results = self.search_relevant_tables(query, top_k)
        if not relevant_results:
            return {"answer": "No relevant information found.", "thinking": "", "sources": [], "context_used": ""}
        context_parts, sources = [], []
        for i, result in enumerate(relevant_results):
            summary = result.get("page_summary", "").strip()
            if summary:
                summary = summary[:200] + "..." if len(summary) > 200 else summary
                context_parts.append(f"Source {i+1}: {summary}")
                if include_table_data and result.get("csv_file_path"):
                    table_data = self.load_table_data(result["csv_file_path"], max_rows=3)
                    if table_data:
                        context_parts.append(f"Table data: {table_data}")
                sources.append({
                    "source_file": result.get("source_file", "Unknown"),
                    "page_number": result.get("page_number", "N/A"),
                    "table_id": result.get("table_id", "Unknown"),
                    "certainty": result.get("certainty", 0.0)
                })
        context = "\n".join(context_parts)
        response = self.generate_answer(query, context)
        result = {"answer": response.answer, "sources": sources, "context_used": context[:500] + "..." if len(context) > 500 else context}
        if show_thinking:
            result["thinking"] = response.thinking
        return result

    def close(self):
        if self.client:
            self.client.close()


def process_pdf(pdf_path: str):
    pages = identify_potential_table_pages(pdf_path)
    if not pages:
        return []

    tables = extract_tables_from_pages(pdf_path, pages)
    if not tables:
        return []

    saved_files = save_dataframes_to_files(tables, pdf_path, OUTPUT_DIR)
    summarizer = BLIPSummarizer(MULTI_MODAL_MODEL)

    page_summaries: Dict[int, str] = {}
    for p in pages:
        img = pdf_page_to_image(pdf_path, p)
        raw_summary = summarizer.summarize_financial_page(img)
        page_summaries[p] = _sanitize_summary(raw_summary)

    metadata = create_simplified_metadata(tables, pdf_path, saved_files, page_summaries)
    metadata = create_summary_embeddings(metadata, EMBEDDING_MODEL)

    client = setup_weaviate_client(WEAVIATE_URL, WEAVIATE_API_KEY)
    create_schema(client)
    store_in_weaviate(client, metadata)
    client.close()

    return [m["table_id"] for m in metadata]
