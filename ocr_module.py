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
    deskew_im = deskew(opencv_image)
    gray = cv.cvtColor(deskew_im, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (7,7),0)

    thresh = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,117,8 )

    

    cv.imwrite("gray.jpg",gray)
    cv.imwrite("thresh.jpg",thresh)
    cv.imwrite("deskew.jpg",deskew_im)


    return deskew_im

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


def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv.cvtColor(newImage, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9, 9), 0)
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 5))
    dilate = cv.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv.findContours(dilate, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv.contourArea, reverse = True)
    for c in contours:
        rect = cv.boundingRect(c)
        x,y,w,h = rect
        cv.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print (len(contours))
    minAreaRect = cv.minAreaRect(largestContour)
    cv.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle
# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv.warpAffine(newImage, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return newImage

# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)