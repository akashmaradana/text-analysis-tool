import streamlit as st
from text_analyzer import TextAnalyzer

# Page configuration
st.set_page_config(
    page_title="Text Analysis Tool",
    page_icon="📝",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .result-box {
            background-color: #f0f2f6;
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        .entity-tag {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            margin: 0.2rem;
            border-radius: 1rem;
            font-size: 0.9rem;
            background-color: #e0e0e0;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_analyzer():
    return TextAnalyzer()

# Title and description
st.title("📝 Text Analysis Tool")
st.markdown("""
This tool provides comprehensive text analysis including:
- Text Summarization
- Sentiment Analysis
- Keyword Extraction
- Named Entity Recognition
""")

# Initialize analyzer
with st.spinner("Loading models..."):
    analyzer = load_analyzer()

# Input section
text = st.text_area(
    "Enter your text for analysis",
    height=200,
    placeholder="Paste your text here (minimum 100 characters)..."
)

if st.button("Analyze Text") and len(text) >= 100:
    with st.spinner("Analyzing text..."):
        results = analyzer.analyze_text(text)
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "Summary", "Sentiment", "Keywords", "Entities"
        ])
        
        # Summary Tab
        with tab1:
            st.markdown("### Summary")
            st.markdown(f"<div class='result-box'>{results['summary']}</div>", unsafe_allow_html=True)
        
        # Sentiment Tab
        with tab2:
            st.markdown("### Sentiment Analysis")
            sentiment = results['sentiment']
            st.markdown(f"<div class='result-box'>")
            st.markdown(f"**Sentiment:** {sentiment['sentiment']}")
            st.progress(sentiment['confidence'])
            st.markdown(f"Confidence: {sentiment['confidence']:.2%}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Keywords Tab
        with tab3:
            st.markdown("### Key Phrases")
            cols = st.columns(5)
            for idx, keyword in enumerate(results['keywords']):
                cols[idx].markdown(f"<div class='result-box' style='text-align: center;'>{keyword}</div>", unsafe_allow_html=True)
        
        # Entities Tab
        with tab4:
            st.markdown("### Named Entities")
            if results['entities']:
                st.markdown(
                    " ".join([f"<span class='entity-tag'>{entity}</span>" for entity in results['entities']]),
                    unsafe_allow_html=True
                )
            else:
                st.info("No named entities found in the text.")

elif len(text) > 0 and len(text) < 100:
    st.error("Please enter at least 100 characters for meaningful analysis.") 