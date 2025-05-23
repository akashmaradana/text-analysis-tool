import streamlit as st
from text_analyzer import TextAnalyzer
import time

# Initialize the text analyzer
@st.cache_resource
def get_analyzer():
    return TextAnalyzer()

def main():
    st.set_page_config(
        page_title="Text Analysis Tool",
        page_icon="🔍",
        layout="wide"
    )
    
    # Header
    st.title("📝 Text Analysis Tool")
    st.markdown("""
    This tool provides comprehensive text analysis including summarization, sentiment analysis,
    keyword extraction, and named entity recognition.
    """)
    
    # Initialize analyzer
    analyzer = get_analyzer()
    
    # Text input
    text_input = st.text_area(
        "Enter your text here (minimum 100 characters):",
        height=200
    )
    
    if st.button("Analyze Text") and text_input:
        if len(text_input) < 100:
            st.error("Please enter at least 100 characters for meaningful analysis.")
            return
            
        with st.spinner("Analyzing text..."):
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Text Stats",
                "📝 Summary",
                "😊 Sentiment",
                "🔑 Keywords",
                "👥 Named Entities"
            ])
            
            # Text Statistics
            with tab1:
                stats = analyzer.get_text_stats(text_input)
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Sentences", stats['num_sentences'])
                with col2:
                    st.metric("Words", stats['num_words'])
                with col3:
                    st.metric("Avg. Sentence Length", f"{stats['avg_sentence_length']:.1f}")
                with col4:
                    st.metric("Characters", stats['num_characters'])
            
            # Summary
            with tab2:
                summary = analyzer.summarize_text(text_input)
                st.markdown("### Text Summary")
                st.write(summary)
            
            # Sentiment Analysis
            with tab3:
                sentiment = analyzer.analyze_sentiment(text_input)
                st.markdown("### Sentiment Analysis")
                
                # Create columns for sentiment display
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sentiment", sentiment['label'])
                with col2:
                    st.metric("Confidence", f"{sentiment['score']*100:.1f}%")
                
                # Add a sentiment emoji
                emoji = "😊" if sentiment['label'] == "POSITIVE" else "😔"
                st.markdown(f"<h1 style='text-align: center;'>{emoji}</h1>", unsafe_allow_html=True)
            
            # Keywords
            with tab4:
                keywords = analyzer.extract_keywords(text_input)
                st.markdown("### Key Terms")
                
                # Display keywords with scores
                for word, score in keywords:
                    st.markdown(f"- **{word}** (relevance: {score:.3f})")
            
            # Named Entities
            with tab5:
                entities = analyzer.extract_entities(text_input)
                st.markdown("### Named Entities")
                
                if entities:
                    # Group entities by label
                    entity_groups = {}
                    for entity in entities:
                        label = entity['label']
                        if label not in entity_groups:
                            entity_groups[label] = []
                        entity_groups[label].append(entity['text'])
                    
                    # Display entities by group
                    for label, items in entity_groups.items():
                        st.markdown(f"**{label}**")
                        st.markdown("- " + "\n- ".join(set(items)))
                else:
                    st.info("No named entities found in the text.")

if __name__ == "__main__":
    main() 