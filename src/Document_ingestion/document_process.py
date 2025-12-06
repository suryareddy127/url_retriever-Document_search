from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from typing import List,Union
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)

class DocumentProcessor:
    """Handles document loading and processing"""
    def __init__(self,chunk_size:int=500,chunk_overlap:int=50):
        """
        Document processing
        args: chunk size:size of the text chunks
              chunk_overlap : overlap b/w chunks
        """
        
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    def load_from_url(self,url:str)->List[Document]:
        "load documents from url"
        loader=WebBaseLoader(url)
        return loader.load()  
    
    def load_from_pdf_directory(self,directory:Union[str,Path])->List[Document]:
        "load documents from pdf"
        loader=PyPDFDirectoryLoader(str(directory))
        return loader.load()
    
    def load_from_txt(self,file_path:Union[str,Path])->List[Document]:
        "load documents from txt"
        loader=TextLoader(str(file_path),encoding="utf-8")
        return loader.load()
    
    def load_from_pdf(self,file_path:Union[str,Path])->List[Document]:
        "load documents from pdf"
        loader=PyPDFLoader(str(file_path))
        return loader.load()
    
    
    def load_documents(self, sources:list[str])->list[Document]:
        """load all documents from urls, pdf, directoryloaders, txt files """
        
        docs:list[Document]=[]
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
            else:
                path = Path(src)
                if path.is_dir():       #pdf directory
                    docs.extend(self.load_from_pdf_directory(path))
                elif path.suffix == ".txt":
                    docs.extend(self.load_from_txt(path))
                elif path.suffix == ".pdf":
                    docs.extend(self.load_from_pdf(path))
                else:
                    raise ValueError(f"Unsupported file type:{src}.use url or .txt files ")
        return docs
    
    
    def split_documents(self,documents:List[Document])->List[Document]:
        """split documents into chunks"""
        
        return self.text_splitter.split_documents(documents)
    
    def process_sources(self,sources:List[str])-> list[Document]:
        """Load documents from sources and split them into chunks."""
        
        docs = self.load_documents(sources)
        return self.split_documents(docs)
    