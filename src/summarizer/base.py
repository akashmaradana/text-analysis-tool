from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseSummarizer(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        self.config = kwargs

    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer"""
        pass

    @abstractmethod
    def summarize(self, text: str) -> str:
        """Generate summary for the input text"""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Return current configuration"""
        return self.config

    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration parameters"""
        self.config.update(new_config) 