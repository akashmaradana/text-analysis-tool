from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from .base_agent import BaseAgent, AgentResponse
from ..config import config

RESEARCH_PROMPT = """You are an expert research assistant. Analyze the following query and provided context to generate a comprehensive and accurate response.

Query: {query}

Context:
{context}

Recent Conversation History:
{history}

Instructions:
1. Analyze the query and context thoroughly
2. Synthesize information from multiple sources
3. Provide specific citations for your claims
4. Highlight any uncertainties or areas needing more research
5. Structure your response clearly with relevant sections

Response:"""

class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Research Agent",
            description="Specialized agent for in-depth research and analysis"
        )
        
        # Initialize LLMs
        self.openai_llm = ChatOpenAI(
            model=config.main_model,
            temperature=0.3
        )
        self.anthropic_llm = ChatAnthropic(
            model=config.anthropic_model,
            temperature=0.3
        )
        
        # Initialize embedding model
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model
        )
        
        # Initialize text splitters
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        
        # Initialize vector store and retriever
        self.vectorstore = Chroma(
            persist_directory=str(config.chroma_persist_dir),
            embedding_function=self.embeddings
        )
        self.docstore = InMemoryStore()
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            parent_splitter=self.parent_splitter,
            child_splitter=self.child_splitter
        )
        
        # Initialize prompt
        self.prompt = ChatPromptTemplate.from_template(RESEARCH_PROMPT)
        
    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(query)
        
        # Get conversation history
        history = self._get_relevant_history()
        
        # Format prompt
        formatted_prompt = self.prompt.format(
            query=query,
            context="\n\n".join([doc.page_content for doc in docs]),
            history="\n".join(history)
        )
        
        # Get responses from both models
        openai_response = await self.openai_llm.ainvoke(formatted_prompt)
        anthropic_response = await self.anthropic_llm.ainvoke(formatted_prompt)
        
        # Combine and analyze responses
        final_response = self._synthesize_responses(
            openai_response.content,
            anthropic_response.content
        )
        
        return AgentResponse(
            content=final_response,
            source_documents=docs,
            metadata={
                "num_sources": len(docs),
                "models_used": ["gpt-4", "claude-3"]
            }
        )
    
    def _synthesize_responses(self, openai_response: str, anthropic_response: str) -> str:
        """Combine and analyze responses from different models"""
        # For now, we'll use a simple combination strategy
        # In a more advanced implementation, we could use another LLM call to
        # analyze and synthesize the responses more intelligently
        combined = (
            "Analysis from multiple AI models:\n\n"
            f"Model 1 Analysis:\n{openai_response}\n\n"
            f"Model 2 Analysis:\n{anthropic_response}"
        )
        return combined
    
    async def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to the research agent's knowledge base"""
        self.retriever.add_documents(documents) 