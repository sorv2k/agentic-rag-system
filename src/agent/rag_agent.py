from typing import List, Dict
import logging
from langchain.schema import Document
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

logger = logging.getLogger(__name__)

class RAGAgent:
    """Agentic RAG system with prompt engineering"""
    
    def __init__(
        self,
        model_name: str = "llama3.2",
        base_url: str = "http://localhost:11434"
    ):
        logger.info(f"Initializing RAG Agent with {model_name}")
        self.llm = Ollama(
            model=model_name,
            base_url=base_url,
            temperature=0.7
        )
        
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant that answers questions based on provided context.

Context:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain enough information, say so
3. Cite specific parts of the context when possible
4. Be concise and clear

Answer:"""
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content
            context_parts.append(f"[Document {i} - {source}]\n{content}\n")
        
        return "\n".join(context_parts)
    
    def generate_answer(
        self,
        question: str,
        retrieved_docs: List[Document]
    ) -> Dict[str, any]:
        """Generate answer using RAG"""
        logger.info(f"Generating answer for: {question}")
        
        # Format context
        context = self.format_context(retrieved_docs)
        
        # Generate answer
        try:
            response = self.chain.invoke({
                "context": context,
                "question": question
            })
            
            answer = response['text']
            
            # Extract sources
            sources = [doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]
            
            result = {
                "question": question,
                "answer": answer,
                "sources": list(set(sources)),
                "num_chunks_used": len(retrieved_docs),
                "relevancy_score": 0.85  # Placeholder
            }
            
            logger.info("Answer generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise