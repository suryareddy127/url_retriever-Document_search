# Rag state for langgraph

from typing import List
from langchain_core.documents import Document   
from pydantic import BaseModel

class RagState(BaseModel):
    """
    Represents the state of our RAG (Retrieval Augmented Generation) system.

    Attributes:
        question: The user's query or question.
        documents: A list of retrieved documents relevant to the question.
        answer: The generated answer based on the question and retrieved documents.
    """
    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""