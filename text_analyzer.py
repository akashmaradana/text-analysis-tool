from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)

class TextAnalyzer:
    def __init__(self):
        self._summarizer = None
        self._sentiment_analyzer = None
        self._nlp = None
    
    @property
    def summarizer(self):
        if self._summarizer is None:
            self._summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        return self._summarizer
    
    @property
    def sentiment_analyzer(self):
        if self._sentiment_analyzer is None:
            self._sentiment_analyzer = pipeline("sentiment-analysis")
        return self._sentiment_analyzer
    
    @property
    def nlp(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load('en_core_web_sm')
            except OSError:
                spacy.cli.download('en_core_web_sm')
                self._nlp = spacy.load('en_core_web_sm')
        return self._nlp
        
    def summarize_text(self, text, max_length=130, min_length=30):
        """Generate a summary of the input text."""
        try:
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            return f"Error in summarization: {str(e)}"
    
    def analyze_sentiment(self, text):
        """Analyze the sentiment of the input text."""
        try:
            result = self.sentiment_analyzer(text)[0]
            return {
                'label': result['label'],
                'score': round(result['score'], 4)
            }
        except Exception as e:
            return {'label': 'ERROR', 'score': 0.0}
    
    def extract_keywords(self, text, num_keywords=10):
        """Extract key phrases from the text using TF-IDF."""
        try:
            # Tokenize and clean text
            stop_words = set(stopwords.words('english'))
            words = word_tokenize(text.lower())
            words = [w for w in words if w.isalnum() and w not in stop_words]
            
            # Create document for TF-IDF
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text])
            
            # Get feature names and scores
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Sort keywords by score
            keywords = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
            keywords.sort(key=lambda x: x[1], reverse=True)
            
            return keywords[:num_keywords]
        except Exception as e:
            return []
    
    def extract_entities(self, text):
        """Extract named entities from the text using spaCy."""
        try:
            doc = self.nlp(text)
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            return entities
        except Exception as e:
            return []
    
    def get_text_stats(self, text):
        """Get basic statistics about the text."""
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            return {
                'num_sentences': len(sentences),
                'num_words': len(words),
                'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
                'num_characters': len(text)
            }
        except Exception as e:
            return {}

    def analyze_text(self, text):
        """Perform comprehensive text analysis"""
        return {
            'summary': self.summarize_text(text),
            'sentiment': self.analyze_sentiment(text),
            'keywords': self.extract_keywords(text),
            'entities': self.extract_entities(text),
            'text_stats': self.get_text_stats(text)
        } 