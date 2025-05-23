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
            
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Text Stats",
            "📝 Summary",
            "😊 Sentiment",
            "🔑 Keywords",
            "👥 Named Entities"
        ])
        
        # Text Statistics (fastest, no model loading required)
        with tab1:
            with st.spinner("Calculating text statistics..."):
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
        
        # Keywords (second fastest)
        with tab4:
            with st.spinner("Extracting keywords..."):
                keywords = analyzer.extract_keywords(text_input)
                st.markdown("### Key Terms")
                for word, score in keywords:
                    st.markdown(f"- **{word}** (relevance: {score:.3f})")
        
        # Named Entities
        with tab5:
            with st.spinner("Identifying named entities..."):
                entities = analyzer.extract_entities(text_input)
                st.markdown("### Named Entities")
                
                if entities:
                    entity_groups = {}
                    for entity in entities:
                        label = entity['label']
                        if label not in entity_groups:
                            entity_groups[label] = []
                        entity_groups[label].append(entity['text'])
                    
                    for label, items in entity_groups.items():
                        st.markdown(f"**{label}**")
                        st.markdown("- " + "\n- ".join(set(items)))
                else:
                    st.info("No named entities found in the text.")
        
        # Sentiment Analysis
        with tab3:
            with st.spinner("Analyzing sentiment..."):
                sentiment = analyzer.analyze_sentiment(text_input)
                st.markdown("### Sentiment Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sentiment", sentiment['label'])
                with col2:
                    st.metric("Confidence", f"{sentiment['score']*100:.1f}%")
                
                emoji = "😊" if sentiment['label'] == "POSITIVE" else "😔"
                st.markdown(f"<h1 style='text-align: center;'>{emoji}</h1>", unsafe_allow_html=True)
        
        # Summary (slowest, load last)
        with tab2:
            with st.spinner("Generating summary..."):
                summary = analyzer.summarize_text(text_input)
                st.markdown("### Text Summary")
                st.write(summary)

if __name__ == "__main__":
    main() 