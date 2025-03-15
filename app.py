import streamlit as st
import cv2
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

# Load spaCy models
nlp = spacy.load('en_core_web_sm')

# Set up OCR
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Adjust path if needed

def extract_text_from_image(image):
    """Extracts text from an image using OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(thresh)
    return text

def extract_text_from_pdf(pdf_bytes):
    """Extracts text from a PDF."""
    images = convert_from_bytes(pdf_bytes)
    all_text = ""
    for image in images:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        text = extract_text_from_image(img_cv)
        all_text += text + "\n"
    return all_text

def analyze_sentiment_spacy(text):
    """Analyzes sentiment using spaCy (basic rule-based approach)."""
    doc = nlp(text)
    positive_words = [token.text for token in doc if token.sentiment > 0]
    negative_words = [token.text for token in doc if token.sentiment < 0]

    if len(positive_words) > len(negative_words):
        return "Positive"
    elif len(negative_words) > len(positive_words):
        return "Negative"
    else:
        return "Neutral"

def summarize_text_spacy(text, length=150):
    """Summarizes text using spaCy."""
    doc = nlp(text)
    stopwords = list(STOP_WORDS)
    keywords = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
    word_frequencies = {}
    for word in keywords:
        if word not in word_frequencies:
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] = word_frequencies[word] / max_frequency
    sentence_tokens = [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies:
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
    select_length = int(len(sentence_tokens) * (length / len(text)))
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    final_summary = [word.text for word in summary]
    return ''.join(final_summary)

st.title("Document Analysis App")

uploaded_files = st.file_uploader("Upload images or PDF", type=["jpg", "png", "pdf"], accept_multiple_files=True)

if uploaded_files:
    all_extracted_text = ""

    for uploaded_file in uploaded_files:
        file_type = uploaded_file.type

        if file_type.startswith("image"):
            image = Image.open(uploaded_file)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            text = extract_text_from_image(image_cv)
            all_extracted_text += text + "\n"

        elif file_type == "application/pdf":
            pdf_bytes = uploaded_file.read()
            text = extract_text_from_pdf(pdf_bytes)
            all_extracted_text += text + "\n"

    if all_extracted_text:
        st.subheader("Extracted Text:")
        st.write(all_extracted_text)

        st.subheader("Sentiment Analysis:")
        sentiment = analyze_sentiment_spacy(all_extracted_text)
        st.write(f"Sentiment: {sentiment}")

        st.subheader("Summary:")
        summary = summarize_text_spacy(all_extracted_text)
        st.write(summary)
    else:
        st.write("No text extracted from the uploaded files.")

import numpy as np