import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.summarizer.abstractive import BartSummarizer
from src.summarizer.extractive import BertExtractiveSummarizer

def main():
    # Example text
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
    
    The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
    
    AI applications include advanced web search engines (e.g., Google), recommendation systems (used by YouTube, Amazon and Netflix), understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Tesla), automated decision-making and competing at the highest level in strategic game systems (such as chess and Go). As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect.
    """

    print("Original text:\n", text)
    print("\n" + "="*80 + "\n")

    # Abstractive summarization
    print("Abstractive Summarization (BART):")
    bart_summarizer = BartSummarizer()
    abstractive_summary = bart_summarizer.generate_summary(text)
    print(abstractive_summary)
    print("\n" + "="*80 + "\n")

    # Extractive summarization
    print("Extractive Summarization (BERT):")
    bert_summarizer = BertExtractiveSummarizer()
    extractive_summary = bert_summarizer.generate_summary(text)
    print(extractive_summary)
    print("\n" + "="*80 + "\n")

    # Evaluate summaries
    reference_summary = "AI is machine intelligence that perceives its environment and takes actions to achieve goals. It includes applications like search engines, recommendation systems, and self-driving cars."
    
    print("Evaluation Metrics:")
    print("\nAbstractive Summary Metrics:")
    abstractive_metrics = bart_summarizer.evaluate(text, reference_summary)
    for metric, score in abstractive_metrics.items():
        print(f"{metric}: {score:.4f}")
    
    print("\nExtractive Summary Metrics:")
    extractive_metrics = bert_summarizer.evaluate(text, reference_summary)
    for metric, score in extractive_metrics.items():
        print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main() 