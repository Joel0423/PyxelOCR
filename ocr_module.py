import cv2 as cv
import pytesseract
from pdf2image import convert_from_bytes
import numpy as np
import shutil
import streamlit as st

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = None

# search for tesseract binary in path
@st.cache_resource
def find_tesseract_binary() -> str:
    return shutil.which("tesseract")

# set tesseract binary path
pytesseract.pytesseract.tesseract_cmd = find_tesseract_binary()
if not pytesseract.pytesseract.tesseract_cmd:
    st.error("Tesseract binary not found in PATH. Please install Tesseract.")

def preprocess_image(image):
    """Preprocess image for better OCR results."""
    # Grayscale, Gaussian blur, Otsu's threshold
    opencv_image = np.array(image)
    gray = cv.cvtColor(opencv_image, cv.COLOR_RGB2GRAY)

    thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,15,4)

    return thresh

def extract_text_from_image(image):
    """Extracts text from an image using OCR."""
    preprocessed = preprocess_image(image)
    text = pytesseract.image_to_string(preprocessed, lang='eng', config='--psm 6')
    return text

def extract_text_from_pdf(pdf_bytes):
    """Extracts text from a PDF."""
    images = convert_from_bytes(pdf_bytes)
    all_text = ""
    for image in images:
        img_pre = preprocess_image(image)
        text = extract_text_from_image(img_pre)
        all_text += text + "\n"
    return all_text 