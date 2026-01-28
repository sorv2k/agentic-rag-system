from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
import shutil
from pathlib import Path
import logging

from src.storage.document_loader import DocumentLoader
from src.storage.vector_store import VectorStore
from src.agent.rag_agent import RAGAgent
from src.utils.logger import setup_logger

# Setup
logger = setup_logger(__name__, os.getenv("LOG_LEVEL", "INFO"))
app = FastAPI(title="Agentic RAG API", version="1.0.0")

# Initialize components
doc_loader = DocumentLoader(
    chunk_size=int(os.getenv("CHUNK_SIZE", 500)),
    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50))
)

vector_store = VectorStore(
    embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
)

rag_agent = RAGAgent(
    model_name=os.getenv("MODEL_NAME", "llama3.2"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)

# Load or create vector store
try:
    vector_store.load_vectorstore()
    logger.info("Loaded existing vector store")
except:
    logger.info("No existing vector store found")

# Request models
class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 4

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list
    num_chunks_used: int

# Endpoints
@app.get("/")
async def root():
    return {"message": "Agentic RAG API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "llama3.2"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Save uploaded file
        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File uploaded: {file.filename}")
        
        # Process document
        chunks = doc_loader.load_document(str(file_path))
        
        # Add to vector store
        if not vector_store.vectorstore:
            vector_store.create_vectorstore(chunks)
        else:
            vector_store.add_documents(chunks)
        
        return {
            "message": "Document processed successfully",
            "filename": file.filename,
            "chunks_created": len(chunks)
        }
    
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system"""
    try:
        if not vector_store.vectorstore:
            raise HTTPException(
                status_code=400,
                detail="No documents uploaded. Please upload documents first."
            )
        
        # Retrieve relevant documents
        retrieved_docs = vector_store.similarity_search(
            request.question,
            k=request.k
        )
        
        # Generate answer
        result = rag_agent.generate_answer(
            request.question,
            retrieved_docs
        )
        
        return QueryResponse(**result)
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/init-from-directory")
async def initialize_from_directory(directory: str):
    """Initialize vector store from a directory of documents"""
    try:
        path = Path(directory)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Directory not found")
        
        chunks = doc_loader.load_directory(directory)
        vector_store.create_vectorstore(chunks)
        
        return {
            "message": "Vector store initialized",
            "total_chunks": len(chunks)
        }
    
    except Exception as e:
        logger.error(f"Error initializing from directory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)