from pathlib import Path
from typing import List
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain.schema import Document

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Load and chunk documents"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Select loader based on file extension
        if path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(path))
        elif path.suffix.lower() == '.txt':
            loader = TextLoader(str(path))
        elif path.suffix.lower() in ['.docx', '.doc']:
            loader = Docx2txtLoader(str(path))
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        logger.info(f"Loading document: {file_path}")
        documents = loader.load()
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {file_path}")
        
        return chunks
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """Load all documents from a directory"""
        path = Path(directory_path)
        all_chunks = []
        
        for file_path in path.glob("*"):
            if file_path.is_file():
                try:
                    chunks = self.load_document(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Total chunks from directory: {len(all_chunks)}")
        return all_chunks