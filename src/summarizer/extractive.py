import torch
from transformers import BertTokenizer, BertModel
from .base import BaseSummarizer
from ..config import get_bert_config
import numpy as np
from typing import List

class BertSummarizer(BaseSummarizer):
    def __init__(self, model_name: str):
        super().__init__(model_name, **get_bert_config())
        self.load_model()

    def load_model(self):
        """Load BERT model and tokenizer"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _get_sentence_embeddings(self, sentences: List[str]) -> torch.Tensor:
        """Get embeddings for each sentence"""
        embeddings = []
        with torch.no_grad():
            for sentence in sentences:
                inputs = self.tokenizer(
                    sentence,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(self.device)
                outputs = self.model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu())
        return torch.stack(embeddings)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def summarize(self, text: str) -> str:
        """Generate summary using extractive method"""
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        if not sentences:
            return ""
        
        if len(sentences) <= self.config["min_length"]:
            return text

        # Get sentence embeddings
        embeddings = self._get_sentence_embeddings(sentences)
        
        # Calculate sentence similarities
        similarities = torch.mm(embeddings, embeddings.transpose(0, 1))
        
        # Normalize similarities
        norm = similarities.norm(p=2, dim=1, keepdim=True)
        normalized_similarities = similarities / norm
        
        # Get sentence scores (importance)
        scores = normalized_similarities.sum(dim=1).cpu().numpy()
        
        # Select top sentences
        num_sentences = min(
            self.config["max_length"],
            max(self.config["min_length"], len(sentences) // 3)
        )
        
        selected_indices = np.argsort(scores)[-num_sentences:]
        selected_indices = sorted(selected_indices)
        
        # Combine selected sentences
        summary = ". ".join([sentences[i] for i in selected_indices])
        if not summary.endswith((".", "!", "?")):
            summary += "."
            
        return summary 