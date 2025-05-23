import torch
from transformers import (
    BartForConditionalGeneration, 
    BartTokenizer,
    pipeline
)
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter

class TextAnalyzer:
    def __init__(self):
        # Initialize models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Initializing NLTK...")
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        # Load summarization model
        print("Loading summarization model...")
        self.sum_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(self.device)
        self.sum_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        
        # Load sentiment analysis
        print("Loading sentiment analysis model...")
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
        print("All models loaded successfully!")

    def generate_summary(self, text, max_length=130, min_length=30):
        """Generate a summary of the input text"""
        inputs = self.sum_tokenizer.encode(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)

        summary_ids = self.sum_model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        summary = self.sum_tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return summary

    def analyze_sentiment(self, text):
        """Analyze the sentiment of the text"""
        result = self.sentiment_analyzer(text)[0]
        return {
            'sentiment': result['label'],
            'confidence': result['score']
        }

    def extract_keywords(self, text, top_n=5):
        """Extract key phrases from the text using NLTK"""
        # Tokenize and get stopwords
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        words = [word for word in words if word.isalpha() and word not in stop_words]
        
        # Get word frequencies
        word_freq = Counter(words)
        
        # Get top N keywords
        keywords = [word for word, _ in word_freq.most_common(top_n)]
        return keywords

    def extract_entities(self, text):
        """Extract basic entities using NLTK"""
        sentences = sent_tokenize(text)
        words = [word_tokenize(sent) for sent in sentences]
        
        # Basic entity extraction (capitalized words)
        entities = []
        for sent in words:
            for word in sent:
                if word[0].isupper() and word.isalpha():
                    entities.append(word)
        
        return list(set(entities))

    def analyze_text(self, text):
        """Perform comprehensive text analysis"""
        return {
            'summary': self.generate_summary(text),
            'sentiment': self.analyze_sentiment(text),
            'keywords': self.extract_keywords(text),
            'entities': self.extract_entities(text)
        } 