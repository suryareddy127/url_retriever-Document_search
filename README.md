# Cognito Weaver (URL Retriever-Document Search): An Agentic RAG System

Cognito Weaver is an intelligent question-answering application that leverages a sophisticated **Retrieval-Augmented Generation (RAG)** pipeline. It dynamically ingests knowledge from user-provided sources (web pages, PDFs, directories) and uses an autonomous agent to reason about the best way to answer a query — either by searching the provided documents or by looking up general knowledge from Wikipedia.

---

## 📖 Overview

This project moves beyond simple RAG workflows by implementing a **stateful, graph-based architecture** with **LangGraph**.  
At its core is a **ReAct (Reasoning and Acting) agent** that functions as the system's brain. Equipped with specialized tools, it intelligently decides which information source to use, resulting in more accurate and comprehensive answers.

The entire system is wrapped in a **user-friendly Streamlit interface**, making it easy for anyone to upload documents and start asking questions.

---

## ✨ Core Features

- **Dynamic Knowledge Base** → Load documents on-the-fly from URLs, PDF files, or entire directories.  
- **Agentic Reasoning** → ReAct agent decides between searching local documents or Wikipedia.  
- **Graph-Based Workflow** → Built with LangGraph for modularity, scalability, and easy debugging.  
- **Semantic Search** → Hugging Face sentence-transformers + FAISS for fast, contextually relevant retrieval.  
- **Interactive UI** → Clean Streamlit interface for easy interaction and configuration.  
- **DeepSeek LLM Integration** → Powered by `deepseek-ai/DeepSeek-V3.2` for robust, context-aware responses. 
---

## 🛠️ How It Works: Architecture

The project follows a clean, multi-step process from data ingestion to answer generation:

1. **Interactive UI & Configuration (`streamlit_app.py`)**  
   - Sidebar allows input of document sources (URLs, file paths).  
   - Clicking *Initialize Pipeline* triggers the backend process (cached for performance).  

2. **Dynamic Document Ingestion (`src/Document_ingestion/document_process.py`)**  
   - Detects source type (URL, PDF, directory).  
   - Uses LangChain loaders to ingest content.  
   - Splits text into overlapping chunks via `RecursiveCharacterTextSplitter`.  

3. **Semantic Vector Store (`src/vectorstore/vectorstore.py`)**  
   - Converts text chunks into embeddings using Hugging Face models.  
   - Indexes embeddings in a FAISS vector store for efficient semantic search.  

4. **Stateful Graph Workflow (`src/graph/graph_builder.py`)**  
   - Defines a `RagState` object carrying the question, retrieved docs, and final answer.  
   - Graph nodes include:  
     - **retriever** → fetches relevant documents.  
     - **response** → generates the final answer using the agent.  

5. **Agentic Reasoning with ReAct (`src/nodes/reactnode.py`)**  
   - The “brain” of the system.  
   - Equipped with tools:  
     - **retriever tool** → semantic search over user-provided documents.  
     - **wikipedia tool** → fallback to general knowledge.  
   - Autonomously decides the best tool(s) for robust, context-aware responses.  

---

## ⚙️ Technologies Used

- **Application Framework** → Streamlit  
- **LLM Orchestration** → LangChain, LangGraph  
- **Agent Framework** → LangChain ReAct  
- **Embeddings** → Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`)
- **LLM Backend** → DeepSeek API (`deepseek-ai/DeepSeek-V3.2`)  
- **Vector Store** → FAISS (Facebook AI Similarity Search)  
- **Document Loading** → LangChain Community Loaders (Web, PDF, Text)  

---

## 📦 Setup & Installation

### Clone the repository
```bash
git clone <repository-url>
cd url_retriever-document_search
uv venv
.venv\Scripts\activate
uv add -r requirements.txt
streamlit run streamlit_app.py



