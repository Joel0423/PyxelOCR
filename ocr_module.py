import cv2 as cv
import pytesseract
from pdf2image import convert_from_bytes
import numpy as np
import shutil
import streamlit as st
import requests
import os

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
    #deskew_im = deskew(opencv_image)
    gray = cv.cvtColor(opencv_image, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (7,7),0)

    thresh = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,117,8 )

    return thresh

def extract_text_from_image(image):
    """Extracts text from an image using OCR."""
    preprocessed = preprocess_image(image)
    cv.imwrite("temp/processed.jpg", preprocessed)

    compress_image("temp/processed.jpg", "temp/compression_output/compressed.jpg", target_size_kb=1024)
    
    response = ocr_space_file("temp/compression_output/compressed.jpg")
    response_json = response.json()
    text = response_json["ParsedResults"][0]["ParsedText"]

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

def ocr_space_file(filename, overlay=False, api_key='d02f75016f88957', language='eng'):
    """ OCR.space API request with local file.
        Python3.5 - not tested on 2.7
    :param filename: Your file path & name.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """

    payload = {'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload,
                          )
    return r

def compress_image(input_path, output_path, target_size_kb=1024, step=5):
    """
    Compress an image to a target file size using OpenCV.
    
    :param input_path: Path to the input image.
    :param output_path: Path to save the compressed image.
    :param target_size_kb: Target file size in KB (default: 1024 KB).
    :param step: Reduction step for JPEG quality (default: 5).
    """
    # Load the image
    img = cv.imread(input_path)
    
    # Set initial quality
    quality = 95  # Start with high quality
    target_size_bytes = target_size_kb * 1024  # Convert KB to Bytes
    
    while quality > 0:
        # Save the image with the current quality
        cv.imwrite(output_path, img, [cv.IMWRITE_JPEG_QUALITY, quality])
        
        # Check file size
        if os.path.getsize(output_path) <= target_size_bytes:
            break  # Stop if within the target size
        
        # Reduce quality for further compression
        quality -= step  
