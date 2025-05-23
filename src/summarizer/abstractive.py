import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from .base import BaseSummarizer
from ..config import get_bart_config

class BartSummarizer(BaseSummarizer):
    def __init__(self, model_name: str):
        super().__init__(model_name, **get_bart_config())
        self.load_model()

    def load_model(self):
        """Load BART model and tokenizer"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def summarize(self, text: str) -> str:
        """Generate summary using BART model"""
        inputs = self.tokenizer.encode(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)

        summary_ids = self.model.generate(
            inputs,
            max_length=self.config["max_length"],
            min_length=self.config["min_length"],
            length_penalty=self.config["length_penalty"],
            num_beams=self.config["num_beams"],
            early_stopping=self.config["early_stopping"]
        )

        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return summary 