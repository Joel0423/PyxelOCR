import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
from ocr_module import extract_text_from_image, extract_text_from_pdf
from sentiment_module import SentimentAnalyzer
from summarizer_module import TextSummarizer

# Initialize the analyzers
sentiment_analyzer = SentimentAnalyzer()
text_summarizer = TextSummarizer()

st.title("Advanced Document Analysis App")

# Sidebar for input method selection
input_method = st.sidebar.radio("Choose Input Method", 
                              ["File Upload", "Camera Capture"])

if input_method == "File Upload":
    uploaded_files = st.file_uploader("Upload images or PDF", 
                                    type=["jpg", "jpeg", "png", "pdf"], 
                                    accept_multiple_files=True)
    
    if uploaded_files:
        all_extracted_text = ""
        
        with st.spinner("Processing files..."):
            for uploaded_file in uploaded_files:
                file_type = uploaded_file.type
                
                if file_type.startswith("image"):
                    image = Image.open(uploaded_file)

                    text = extract_text_from_image(image)
                    all_extracted_text += text + "\n"
                
                elif file_type == "application/pdf":
                    pdf_bytes = uploaded_file.read()
                    text = extract_text_from_pdf(pdf_bytes)
                    all_extracted_text += text + "\n"

else:  # Camera Capture
    st.write("Camera Input")
    camera_image = st.camera_input("Take a picture")
    
    if camera_image is not None:
        # Process the camera image
        image = Image.open(camera_image)
        all_extracted_text = extract_text_from_image(image)

# Process and display results if we have extracted text
if 'all_extracted_text' in locals() and all_extracted_text:
    st.subheader("Extracted Text:")
    with st.expander("Show Extracted Text"):
        st.write(all_extracted_text)
    
    # Sentiment Analysis
    st.subheader("Sentiment Analysis:")
    with st.spinner("Analyzing sentiment..."):
        sentiment_results = sentiment_analyzer.analyze_sentiment_combined(all_extracted_text)
        st.write(f"Overall Sentiment: {sentiment_results['sentiment']}")
        st.write(f"Sentiment Score: {sentiment_results['score']:.2f}")
        
        # Display emotional tone
        emotions = sentiment_analyzer.get_emotional_tone(all_extracted_text)
        st.write("Emotional Tone Analysis:")
        for emotion, count in emotions.items():
            if count > 0:
                st.write(f"- {emotion.capitalize()}: {count} instances")
    
    # Text Summarization
    st.subheader("Text Summary:")
    summary_type = st.radio("Choose summarization method:", 
                          ["Extractive", "Abstractive", "Hybrid"])
    
    with st.spinner("Generating summary..."):
        if summary_type == "Extractive":
            summary = text_summarizer.extractive_summarize(all_extracted_text)
            st.write(summary)
        elif summary_type == "Abstractive":
            summary = text_summarizer.abstractive_summarize(all_extracted_text)
            st.write(summary)
        else:  # Hybrid
            summary = text_summarizer.hybrid_summarize(all_extracted_text)
            st.write("Final Summary:")
            st.write(summary['final_summary'])
            with st.expander("Show Extractive Summary"):
                st.write(summary['extractive_summary'])
    
    # Key Phrases
    st.subheader("Key Phrases:")
    with st.spinner("Extracting key phrases..."):
        key_phrases = text_summarizer.get_key_phrases(all_extracted_text)
        st.write(", ".join(key_phrases))

else:
    st.write("Please upload files or take a picture to analyze.")