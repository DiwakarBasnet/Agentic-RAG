# Financial RAG

## Overview

This RAG pipeline processes PDF documents, extracts table and textual data and then summarizes the page contents using a multimodal model so that the summaries for the table are stored in the table metadata.
These data are stored in a Weaviate vector database, and then uses an LLM-based retrieval system to answer natural language queries using the stored knowledge.

## Getting Started

### Prerequisites

* Python 3.10 or higher
* [Conda](https://docs.conda.io/en/latest/) (for environment management)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/DiwakarBasnet/Agentic-RAG.git
   cd Agentic_RAG
   ```

2. **Create and Activate the Conda Environment**

   ```bash
   conda env create -f environment.yml
   conda activate rag
   ```

3. **Install Additional Dependencies**

   ```bash
   pip install -r requirements.txt
   ```
---

## Environment Variables and API Keys to include
* Huggingface API Token
* Weaviate API Key
* Weaviate URL (URL for our weaviate cluster)
* Weaviate Class Name

---

## High level working

[!Architecture Diagram](assets/pipeline_diagram.png)
