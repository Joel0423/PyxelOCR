import cv2 as cv
from pdf2image import convert_from_bytes
import numpy as np
import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def preprocess_image(image):
    
    # Grayscale, Gaussian blur, adaptive threshold
    opencv_image = np.array(image)
    #deskew_im = deskew(opencv_image)
    gray = cv.cvtColor(opencv_image, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (7,7),0)

    thresh = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,117,8 )

    return thresh

def extract_text_from_image(image):
    API_KEY = os.getenv("API_KEY")
    
    preprocessed = preprocess_image(image)
    cv.imwrite(f"temp/processed/{st.session_state.user_id}_processed.jpg", preprocessed)

    compress_image(f"temp/processed/{st.session_state.user_id}_processed.jpg", f"temp/compression_output/{st.session_state.user_id}_compressed.jpg", target_size_kb=1024)
    
    response = ocr_space_file(f"temp/compression_output/{st.session_state.user_id}_compressed.jpg", api_key=API_KEY)
    response_json = response.json()

    # Variables to store the line with largest MaxHeight
    max_height_line = None
    max_height = 0

    # Parse through the results
    for result in response_json["ParsedResults"]:
        if result["TextOverlay"] and result["TextOverlay"]["Lines"]:
            for line in result["TextOverlay"]["Lines"]:
                # Check for largest MaxHeight
                if line["MaxHeight"] > max_height:
                    max_height = line["MaxHeight"]
                    max_height_line = line
                    line_text = " ".join(word["WordText"] for word in max_height_line["Words"])

    text = response_json["ParsedResults"][0]["ParsedText"]

    return line_text, text

def extract_text_from_pdf(pdf_bytes):
    
    images = convert_from_bytes(pdf_bytes)
    all_text = ""
    for image in images:
        img_pre = preprocess_image(image)
        text = extract_text_from_image(img_pre)
        all_text += text + "\n"
    return all_text 

def ocr_space_file(filename, overlay=True, api_key='helloworld', language='eng'):
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
    st.write(api_key)

    payload = {'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload
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
