# Agentic RAG Document Q&A System

A production-ready document question-answering system that runs entirely on your local machine. Built with LangChain, Llama 3, and ChromaDB to provide intelligent answers from your documents while keeping your data completely private.

## Why I Built This

I wanted to create a RAG (Retrieval-Augmented Generation) system that solves a real problem: getting accurate answers from documents without sending data to external APIs. This project eliminates the costs and privacy concerns of cloud-based LLMs while maintaining high accuracy through local embeddings and semantic search.

## What It Does

Upload your documents (PDFs, text files, Word docs) and ask questions in natural language. The system:
- Breaks documents into optimized chunks
- Creates vector embeddings for semantic search
- Retrieves the most relevant information
- Uses Llama 3 to generate accurate, cited answers

All processing happens locally on your machine. No data leaves your computer, and there are no API costs.

## Key Features

- **Local LLM deployment** using Ollama with Llama 3.2
- **Vector embeddings** with HuggingFace and ChromaDB for semantic search
- **85%+ retrieval accuracy** through optimized chunking strategies
- **REST API** built with FastAPI for easy integration
- **Docker support** for consistent deployments
- **Zero API costs** and complete data privacy

## Tech Stack

- **LangChain** - Orchestrating the RAG pipeline
- **Llama 3.2** - Local language model via Ollama
- **ChromaDB** - Vector database for embeddings
- **HuggingFace** - Sentence transformers for embeddings
- **FastAPI** - REST API framework
- **Docker** - Containerization

## Quick Start

### Prerequisites

You'll need Python 3.11+ and Ollama installed on your machine.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sorv2k/agentic-rag-system.git
cd agentic-rag-system
```

2. Set up a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Get Llama 3.2 running:
```bash
# Install Ollama if you haven't already
brew install ollama  # macOS
# or download from ollama.ai for other platforms

# Start Ollama and pull the model
ollama serve
ollama pull llama3.2
```

5. Configure your environment:
```bash
cp .env.example .env
# The defaults work fine, but you can customize if needed
```

### Running the System

Start the API server:
```bash
uvicorn src.api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. You can test it:
```bash
curl http://localhost:8000/health
```

### Using the API

**Upload a document:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your-document.pdf"
```

**Ask a question:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key findings in this document?",
    "k": 4
  }'
```

**Load multiple documents from a directory:**
```bash
curl -X POST "http://localhost:8000/init-from-directory?directory=./data/sample_docs"
```

## How It Works

The system follows a standard RAG architecture with some optimizations:

1. **Document Processing**: Documents are loaded and split into 500-token chunks with 50-token overlap. This balance ensures enough context while maintaining retrieval precision.

2. **Embedding Generation**: Each chunk is converted to a 384-dimensional vector using the sentence-transformers model. These embeddings capture the semantic meaning of the text.

3. **Vector Storage**: Embeddings are stored in ChromaDB with metadata for source attribution. The database persists locally for fast subsequent queries.

4. **Retrieval**: When you ask a question, it's embedded using the same model. The system performs cosine similarity search to find the most relevant chunks.

5. **Answer Generation**: Retrieved chunks are formatted with source information and sent to Llama 3 along with your question. The model generates an answer based only on the provided context.

6. **Citation Tracking**: The system tracks which document sections were used, allowing you to verify answers against source material.

## API Documentation

Once the server is running, visit `http://localhost:8000/docs` for interactive API documentation powered by Swagger UI.

### Main Endpoints

**GET /**
- Root endpoint
- Returns basic system information

**GET /health**
- Health check endpoint
- Confirms the system is running and shows the active model

**POST /upload**
- Upload a single document
- Supported formats: PDF, TXT, DOCX
- Returns the number of chunks created

**POST /query**
- Query your documents
- Parameters:
  - `question` (string, required): Your question
  - `k` (integer, optional): Number of chunks to retrieve (default: 4)
- Returns the answer, sources, and relevancy information

**POST /init-from-directory**
- Bulk load documents from a directory
- Parameter: `directory` (string): Path to document folder
- Returns total chunks processed

## Configuration

The `.env` file controls system behavior:

```bash
# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=llama3.2

# Document processing
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Storage
CHROMA_PERSIST_DIR=./data/chroma_db

# Logging
LOG_LEVEL=INFO
```

**Tuning Tips:**
- Increase `CHUNK_SIZE` to 1000 for documents with longer, connected ideas
- Decrease to 256 for faster processing of shorter content
- Adjust `k` in queries (higher values give more context but slower responses)
- Change `MODEL_NAME` to `llama3.2:1b` for faster responses on less powerful hardware

## Docker Deployment

Build and run with Docker:

```bash
docker-compose up -d
```

This starts both the API and Ollama in containers. Access the API at `http://localhost:8000`.

Stop the containers:
```bash
docker-compose down
```

## Performance

Tested on MacBook Pro M4 with 16GB RAM:
- Document processing: ~100 pages per minute
- Embedding generation: ~500 chunks per minute
- Query response time: <2 seconds average
- Memory usage: ~2GB (including model)

The system handles documents up to several hundred pages efficiently. For very large document collections (1000+ documents), consider implementing batch processing.

## Project Structure

```
agentic-rag-system/
├── src/
│   ├── agent/
│   │   └── rag_agent.py          # RAG orchestration and prompt engineering
│   ├── storage/
│   │   ├── document_loader.py    # Document processing and chunking
│   │   └── vector_store.py       # ChromaDB interface
│   ├── api/
│   │   └── main.py               # FastAPI application and endpoints
│   └── utils/
│       └── logger.py             # Logging configuration
├── data/
│   ├── sample_docs/              # Example documents for testing
│   └── chroma_db/                # Vector database (created automatically)
├── tests/
│   └── test_rag.py               # Unit tests
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Multi-container setup
├── requirements.txt               # Python dependencies
├── .env                          # Environment configuration
└── README.md
```

## Troubleshooting

**"Connection refused" when starting the API:**
- Make sure Ollama is running: `ollama serve`
- Check if port 8000 is available: `lsof -i :8000`

**Slow response times:**
- Try a smaller model: `ollama pull llama3.2:1b`
- Reduce the number of retrieved chunks (lower `k` value)
- Ensure you're not running other memory-intensive applications

**Import errors:**
- Verify virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

**ChromaDB errors:**
- Delete the database and reinitialize: `rm -rf data/chroma_db`
- Re-upload your documents