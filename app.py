import streamlit as st
from PIL import Image
import os
from datetime import datetime
import json
import install
install.install()

from ocr_module import extract_text_from_image, extract_text_from_pdf
from sentiment_module import SentimentAnalyzer
from summarizer_module import TextSummarizer
from clean_txt_module import clean_text
from streamlit_image_select import image_select
from browser_detection import browser_detection_engine


# Create necessary folders if they don't exist
required_folders = [
    'temp',
    'temp/carousel',
    'temp/compression_output',
    'temp/processed',
    'user_outputs'
]



for folder in required_folders:
    os.makedirs(folder, exist_ok=True)

# Initialize session state for user ID, outputs if not exists
if 'user_id' not in st.session_state:
    st.session_state.user_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if 'outputs' not in st.session_state:
    st.session_state.outputs = 0

# Initialize the analyzers
sentiment_analyzer = SentimentAnalyzer()
text_summarizer = TextSummarizer()
browser_value = browser_detection_engine()

def save_analysis_output(text, sentiment_results, emotions, summary, key_phrases, large_line):
    """Save analysis results to a file."""
    # clean the text of nonsensical words (ocr errors)
    text = clean_text(text)
    summary = clean_text(summary)


    # Create title from first few words
    if large_line:
        large_line = clean_text(large_line)
        large_line = large_line.split(' ')
        title = " ".join(large_line[:min(8, len(large_line))])

    else:
        
        title = ' '.join(text.split()[:5]) + "..."
    
    # Create output directory for user if it doesn't exist
    user_dir = os.path.join('user_outputs', st.session_state.user_id)
    os.makedirs(user_dir, exist_ok=True)
    
    # Create unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_{timestamp}.json"
    filepath = os.path.join(user_dir, filename)
    
    # Prepare output data
    output_data = {
        'title': title,
        'timestamp': timestamp,
        'extracted_text': text,
        'sentiment': sentiment_results,
        'emotions': emotions,
        'summary': summary,
        'key_phrases': key_phrases
    }
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    return title, filepath

def load_historical_outputs():
    """Load all historical outputs for the current user."""
    user_dir = os.path.join('user_outputs', st.session_state.user_id)
    if not os.path.exists(user_dir):
        return []
    
    outputs = []
    for filename in os.listdir(user_dir):
        if filename.endswith('.json'):
            with open(os.path.join(user_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                outputs.append(data)
    
    return sorted(outputs, key=lambda x: x['timestamp'], reverse=True)

def display_analysis_output(output_data):
    """Display the analysis output in the main area."""
    st.title(output_data["title"])
    st.subheader("Analysis Results")
    
    with st.expander("Extracted Text"):
        st.write(output_data['extracted_text'])
    
    st.subheader("Sentiment Analysis")
    st.write(f"Overall Sentiment: {output_data['sentiment']['sentiment']}")
    st.write(f"Sentiment Score: {output_data['sentiment']['score']:.2f}")
    st.write(f"Subjectivity: {output_data['sentiment']['subjectivity']:.2f}")
    
    st.write("Emotional Tone Analysis:")
    for emotion, count in output_data['emotions'].items():
        if count > 0:
            st.write(f"- {emotion.capitalize()}: {count} instances")
    
    st.subheader("Summary")
    st.write(output_data['summary'])
    
    st.subheader("Key Phrases")
    st.write(", ".join(output_data['key_phrases']))

def show_file_upload():
    # File upload and processing
    st.subheader("High resolution images give better results\nKeep the text as straight as possible")
    uploaded_files = st.file_uploader("Upload images or PDF", 
                                    type=["jpg", "jpeg", "png", "pdf"], 
                                    accept_multiple_files=True)
    
    if uploaded_files:
        # Create temporary image files for image_select
        image_paths = []
        for i, uploaded_file in enumerate(uploaded_files):
            if uploaded_file.type.startswith("image"):
                image = Image.open(uploaded_file)
                
                if browser_value['isMobile']:
                    image = image.rotate(270, expand=True)
                # Save image temporarily
                temp_path = f"temp/carousel/temp_{st.session_state.user_id}_{st.session_state.outputs}_{i}.jpg"
                image.save(temp_path)
                image_paths.append(temp_path)
        
        if image_paths:

            # Display image selector
            selected_image_index = image_select(
                "Select a document to view",
                image_paths,
                use_container_width=True,
                return_value="index"
            )
            
            # Show selected image details
            if selected_image_index is not None:
                st.image(image_paths[selected_image_index], width=600)

            # Clean up temporary files
            for path in image_paths:
                if os.path.exists(path):
                    os.remove(path)
            all_extracted_text = ""
        
        if st.button("Extract Text"):    
            
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    file_type = uploaded_file.type
                    
                    if file_type.startswith("image"):
                        image = Image.open(uploaded_file)
                        if browser_value['isMobile']:
                            image = image.rotate(270, expand=True)
                        large_line, text = extract_text_from_image(image)
                        all_extracted_text += text + "\n"
                    
                    elif file_type == "application/pdf":
                        pdf_bytes = uploaded_file.read()
                        text = extract_text_from_pdf(pdf_bytes)
                        all_extracted_text += text + "\n"
            
            if all_extracted_text:
                with st.spinner("Analyzing text..."):
                    # Perform analysis
                    sentiment_results = sentiment_analyzer.analyze_sentiment_combined(all_extracted_text)
                    emotions = sentiment_analyzer.get_emotional_tone(all_extracted_text)
                    summary = text_summarizer.hybrid_summarize(all_extracted_text)['final_summary']
                    key_phrases = text_summarizer.get_key_phrases(all_extracted_text)
                    
                    # Save results
                    title, filepath = save_analysis_output(
                        all_extracted_text,
                        sentiment_results,
                        emotions,
                        summary,
                        key_phrases,
                        large_line
                    )
                    
                    st.success(f"Analysis saved as: {title}")
                    st.session_state.outputs = st.session_state.outputs+1
                    historical_outputs = load_historical_outputs()

                    if historical_outputs:
                        for output in historical_outputs:
                            if output['title'] == title:
                                st.session_state.selected_output = output
                                st.rerun()
            else:
                st.warning("No text was extracted from the uploaded files.")

# Main app layout
st.title("PyxelOCR: Document Analysis App")

# Sidebar for historical outputs
st.sidebar.title("== History ==")
historical_outputs = load_historical_outputs()

if historical_outputs:
    st.sidebar.write("Click on any analysis to view details:")
    for output in historical_outputs:
        if st.sidebar.button(output['title'], key=output['timestamp']):
            st.session_state.selected_output = output
            st.rerun()

# Main content area
if 'selected_output' in st.session_state:
    back = st.button("Go Back")
    if back:
        del st.session_state['selected_output']
        st.rerun()
    display_analysis_output(st.session_state.selected_output)
else:
    show_file_upload()

