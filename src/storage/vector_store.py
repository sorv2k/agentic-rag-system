from typing import List, Optional
import logging
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

class VectorStore:
    """Manage vector embeddings with ChromaDB"""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "./data/chroma_db"
    ):
        logger.info(f"Initializing embeddings with {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}  # Use 'mps' for M4 GPU acceleration
        )
        self.persist_directory = persist_directory
        self.vectorstore: Optional[Chroma] = None
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Create vector store from documents"""
        logger.info(f"Creating vector store with {len(documents)} documents")
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        logger.info("Vector store created successfully")
        return self.vectorstore
    
    def load_vectorstore(self) -> Chroma:
        """Load existing vector store"""
        logger.info("Loading existing vector store")
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        
        return self.vectorstore
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        logger.info(f"Searching for: {query}")
        results = self.vectorstore.similarity_search(query, k=k)
        logger.info(f"Found {len(results)} relevant chunks")
        
        return results
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to existing vector store"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        self.vectorstore.add_documents(documents)