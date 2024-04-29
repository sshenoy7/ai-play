import cv2
import pytesseract
from transformers import pipeline

# Load pre-trained GPT-2 model for text generation
text_similarity_checker = pipeline("text-similarity", model="textattack/bert-base-uncased-STS-B")

# Function to extract text from an image using OCR
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

# Function to compare two images based on their text representations
def compare_images(image_path1, image_path2):
    text1 = extract_text_from_image(image_path1)
    text2 = extract_text_from_image(image_path2)
    similarity_score = text_similarity_checker(text1, text2)[0]['score']
    return similarity_score

# Paths to the images to be compared
image_path1 = "V.png"
image_path2 = "LJ1.png"

# Compare the images and get the similarity score
similarity_percentage = compare_images(image_path1, image_path2)

print(f"Similarity Percentage: {similarity_percentage}")
