from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from .base_agent import BaseAgent, AgentResponse
from ..config import config

QA_PROMPT = """You are an expert question answering assistant. Your goal is to provide accurate, concise answers based on the provided context.

Question: {query}

Context:
{context}

Recent Questions:
{history}

Instructions:
1. Answer the question directly and concisely
2. Only use information from the provided context
3. If the answer is not in the context, say so
4. Cite specific sources for your answer
5. Express your confidence level in the answer

Answer:"""

class QAAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="QA Agent",
            description="Specialized agent for precise question answering"
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.main_model,
            temperature=0.1  # Lower temperature for more precise answers
        )
        
        # Initialize embedding model
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model
        )
        
        # Initialize vector store
        self.vectorstore = Chroma(
            persist_directory=str(config.chroma_persist_dir),
            embedding_function=self.embeddings
        )
        
        # Initialize multi-query retriever
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 4}
            ),
            llm=self.llm
        )
        
        # Initialize prompt
        self.prompt = ChatPromptTemplate.from_template(QA_PROMPT)
    
    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        # Generate multiple query variations and retrieve relevant documents
        docs = await self.retriever.aget_relevant_documents(query)
        
        # Get conversation history
        history = self._get_relevant_history()
        
        # Format prompt
        formatted_prompt = self.prompt.format(
            query=query,
            context="\n\n".join([doc.page_content for doc in docs]),
            history="\n".join(history)
        )
        
        # Get response
        response = await self.llm.ainvoke(formatted_prompt)
        
        # Calculate confidence score based on document relevance
        confidence = self._calculate_confidence(query, docs)
        
        return AgentResponse(
            content=response.content,
            confidence=confidence,
            source_documents=docs,
            metadata={
                "num_sources": len(docs),
                "query_variations": len(self.retriever.generate_queries(query))
            }
        )
    
    def _calculate_confidence(self, query: str, docs: List[Document]) -> float:
        """Calculate confidence score based on document relevance"""
        if not docs:
            return 0.0
            
        # Simple heuristic based on number of relevant documents
        # In a more advanced implementation, we could use semantic similarity
        # between the query and documents
        confidence = min(len(docs) / 4, 1.0)
        return confidence
    
    async def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to the QA agent's knowledge base"""
        self.vectorstore.add_documents(documents) 