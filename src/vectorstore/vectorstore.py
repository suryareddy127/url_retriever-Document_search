
from typing import List
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class VectorStore: 
    """ manages vector store application"""
    def __init__(self,model_name:str="sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings=HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore=None
        self.retriever=None
        
    def create_retriever(self,documents:List[Document]):
        "creating vector store from documents"
        
        self.vectorstore=FAISS.from_documents(documents,self.embeddings)
        self.retriever=self.vectorstore.as_retriever()
        
    def get_retriever(self):
        "return the retriever"
        
        if self.retriever is None:
            raise ValueError("retriever is not initialized")
        return self.retriever
    

    def retreive(self,query,k:int=4)->List[Document]: 
        "retrieve documents for a query"
        
        if self.vectorstore is None:
            raise ValueError("vectorstore not initialized")
        return self.retriever.invoke(query)
    
    