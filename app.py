from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import pytesseract
import os
import time
import cv2
import re
import numpy as np
import logging
from logging.handlers import RotatingFileHandler


# Optional: For card name validation
# from mtgjson import CardDatabase
# from fuzzywuzzy import process, fuzz

# Configure logging BEFORE creating Flask app
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('ocr_debug.log', maxBytes=1024000, backupCount=10),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Configure Tesseract path (adjust for your system)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # macOS
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows

# Create upload directory if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def crop(img):
    # Crop to title area (adjust coordinates based on your card images)
    #title_roi = img[50:150, 30:400]  # y:y+h, x:x+w
    title_roi = img  # y:y+h, x:x+w
    return title_roi

def lab_color_space(img):
    # Convert to LAB color space for better contrast control
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    return l_channel, a, b

def clah(img):
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    processed_l = clahe.apply(img)
    return processed_l

def merge_channels(img, a, b):
    # Merge channels and convert back to BGR
    #processed_lab = cv2.merge((processed_l, a, b))
    processed_lab = cv2.merge((img, a, b))
    processed_bgr = cv2.cvtColor(processed_lab, cv2.COLOR_LAB2BGR)
    return processed_bgr

def grey_scale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (3,3), 0)
    # _, thresh = cv2.threshold(blurred, 0, 255, 
    #                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # return thresh

def clean_ocr_output(text):
    """Post-process OCR results for MTG card names"""
    # Remove non-alphanumeric characters except allowed ones
    cleaned = re.sub(r'[^A-Za-z0-9\-\'’ ,.!]', '', text)
    
    # Common MTG-specific corrections
    replacements = {
        'O': '0',  # Zero vs letter O
        'l': '1',  # One vs lowercase L
        'B': '8',  # Eight vs letter B
        '|': 'I',  # Pipe vs letter I
        '‘': "'"   # Smart quotes to standard
    }
    
    for wrong, correct in replacements.items():
        cleaned = cleaned.replace(wrong, correct)
    
    return cleaned.strip()

# Add this function to save preprocessing images
def save_debug_image(image, stage_name):
    debug_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, f"{int(time.time())}_{stage_name}.png")
    cv2.imwrite(path, image)
    app.logger.debug(f"Saved debug image: {path}")
    return path

@app.route('/upload', methods=['POST'])
def upload():
    try:
        app.logger.info("Upload request received")
        
        if 'file' not in request.files:
            app.logger.warning("No file part in request")
            return 'No file uploaded', 400

        file = request.files['file']
        if file.filename == '':
            app.logger.warning("Empty filename submitted")
            return 'No file selected', 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        app.logger.debug(f"File saved to: {filepath}")

        # Preprocessing debug
        raw_img = cv2.imread(filepath)
        save_debug_image(raw_img, "00_raw")

        # process image
        img = raw_img
        #img = crop(img)
        #img, a, b = lab_color_space(img)
        #img = clah(img)
        #img = merge_channels(img, a, b)
        img = grey_scale(img)
        preprocessed_img = img
        dbg_img = save_debug_image(preprocessed_img, "01_preprocessed")
        raw_text = pytesseract.image_to_string(preprocessed_img)

        # OCR debug
        #app.logger.debug(f"Tesseract config: {custom_config}")
        #raw_text = pytesseract.image_to_string(preprocessed_img, config=custom_config)
        #raw_text = pytesseract.image_to_string(preprocessed_img)
        app.logger.info(f"Raw OCR output: {repr(raw_text)}")

        # Post-processing debug
        clean_text = clean_ocr_output(raw_text)
        app.logger.info(f"Cleaned text: {repr(clean_text)}")

        return render_template('result.html', text=raw_text, image_path=dbg_img)

    except pytesseract.TesseractNotFoundError as e:
        app.logger.critical(f"Tesseract error: {str(e)}")
        return "OCR engine not configured properly", 500
        
    except Exception as e:
        app.logger.error(f"Processing failed: {str(e)}", exc_info=True)
        return "An error occurred during processing", 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True)

