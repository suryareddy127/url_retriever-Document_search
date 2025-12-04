import streamlit as st
import time
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

# Import project modules
from src.config.config import config
from src.Document_ingestion.document_process import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph.graph_builder import GraphBuilder


# -------------------------------
# Session State Initialization
# -------------------------------
def init_session_state():
    """Initialize all required session state keys."""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "rag_graph" not in st.session_state:
        st.session_state.rag_graph = None
    if "messages" not in st.session_state:
        st.session_state.messages = []


# -------------------------------
# RAG Pipeline Initialization
# -------------------------------
@st.cache_resource
def initialize_rag_pipeline(sources):
    with st.spinner("Initializing RAG Pipeline... This may take a moment."):
        llm = config.get_llm()

        # Document Processing
        doc_processor = DocumentProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        vector_store = VectorStore()

        # Process sources
        docs = doc_processor.process_url(sources)
        split_docs = doc_processor.split_documents(docs)

        if not split_docs:
            st.warning("No documents were loaded. The retriever will be empty.")
            return None, 0

        # FIXED: use create_retriever
        vector_store.create_retriever(split_docs)
        retriever = vector_store.get_retirever()

        # Build Graph
        graph_builder = GraphBuilder(retriever=retriever, llm=llm)
        rag_graph = graph_builder.build()

    return rag_graph, len(docs)


# -------------------------------
# Streamlit App Configuration
# -------------------------------
st.set_page_config(
    page_title="📄 URL RETRIEVER DOCUMENT SEARCH",
    page_icon="🌐",
    layout="centered",
)

# Initialize session state
init_session_state()


# -------------------------------
# Sidebar UI
# -------------------------------
with st.sidebar:
    st.header("Configuration")

    sources_input = st.text_area(
        "Enter document sources (URLs, PDF folder path), one per line:",
        value=(
            "https://lilianweng.github.io/posts/2023-06-23-agent/\n"
        ),
        height=150,
    )
    sources = [src.strip() for src in sources_input.split("\n") if src.strip()]

    if st.button("Initialize Pipeline"):
        rag_graph, doc_count = initialize_rag_pipeline(sources)
        st.session_state.rag_graph = rag_graph
        st.session_state.doc_count = doc_count
        st.session_state.messages = []
        st.session_state.initialized = True

        if rag_graph:
            st.success(f"Pipeline initialized successfully! Loaded {doc_count} documents.")
        else:
            st.error("Pipeline initialization failed. Please check sources.")


# -------------------------------
# Main Chat Interface
# -------------------------------
st.title("📄 URL Retriever Document Search")
st.markdown("Ask questions about the documents you provided in the sidebar.")

if st.session_state.rag_graph:
    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Handle new user input
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                start_time = time.time()
                result = st.session_state.rag_graph.invoke({"question": prompt})
                end_time = time.time()

            answer = result.get("answer", "Sorry, I couldn't find an answer.")
            generation_time = end_time - start_time
            full_response = f"{answer}\n\n*Answer generated in {generation_time:.2f} seconds.*"
            message_placeholder.markdown(full_response)

            # Display retrieved document sources
            retrieved_docs = result.get("retrieved_docs", [])
            if retrieved_docs:
                with st.expander("📚 View Retrieved Context", expanded=True):
                    for i, doc in enumerate(retrieved_docs):
                        source = doc.metadata.get("source", "Unknown Source")
                        st.info(f"**Source {i+1}:** `{source}`")
                        st.markdown(f"> {doc.page_content}")
            else:
                st.warning("No specific context was retrieved to generate this answer.")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("👋 Welcome! Please enter document sources in the sidebar and click 'Initialize Pipeline' to start.")


# -------------------------------
# Main Function (Optional)
# -------------------------------
def main():
    """Entry point for running outside Streamlit."""
    pass


if __name__ == "__main__":
    main()