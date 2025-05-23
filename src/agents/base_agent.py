from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

class AgentResponse(BaseModel):
    """Structured response from an agent"""
    content: str
    confidence: float = 1.0
    source_documents: Optional[List[Document]] = None
    metadata: Dict[str, Any] = {}

class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
    @abstractmethod
    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process a query and return a response"""
        pass
    
    def _format_source_documents(self, docs: List[Document]) -> str:
        """Format source documents for citation"""
        if not docs:
            return ""
        
        formatted_sources = []
        for i, doc in enumerate(docs, 1):
            source = f"[{i}] "
            if doc.metadata.get("title"):
                source += f"{doc.metadata['title']}"
            if doc.metadata.get("source"):
                source += f" ({doc.metadata['source']})"
            formatted_sources.append(source)
            
        return "\n".join(formatted_sources)
    
    def _get_relevant_history(self, k: int = 5) -> List[str]:
        """Get the k most recent conversation turns"""
        history = self.memory.chat_memory.messages
        return [str(msg) for msg in history[-k:]]
    
    def clear_memory(self) -> None:
        """Clear the agent's conversation memory"""
        self.memory.clear() 