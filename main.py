from flask import Flask, request, render_template
import cv2
import numpy as np
import pytesseract
from tensorflow.keras.models import load_model
import requests
import os
import json

app = Flask(__name__)

# ------------------- Dummy LLM function (for fallback/testing) -------------------
def llm_response(query: str, model_output: str) -> str:
    return f"LLM processed query: '{query}' with model output: '{model_output}'"

# ------------------- Load your CNN models -------------------
# Replace these paths with the actual file paths to your models.
skin_model = load_model('skin-cancer-isic-9-classes_VGG19_V1_ph1_model.h5')
xray_model = load_model('CNN_model.h5')

# Skin disease label mapping.
skin_label_mapping = {
    0: "Acitinic Keratosis",
    1: "Pigmented Benign Keratosis",
    2: "Melanoma",
    3: "Seborrheic Keratosis",
    4: "Basal Cell Carcinoma",
    5: "Nevus",
    6: "Squamous Cell Carcinoma",
    7: "Dermatofibroma",
    8: "Vascular Lesion"
}

# X-ray label mapping for binary classification.
xray_label_mapping = {
    0: "normal",
    1: "pneumoni"
}

# ------------------- Preprocessing Functions -------------------
def preprocess_image(image: np.ndarray, target_size: tuple, grayscale: bool = False) -> np.ndarray:
    """
    Resize the input image to the given target_size and normalize the pixel values.
    
    Args:
        image (np.ndarray): The input image.
        target_size (tuple): The desired output size (width, height).
        grayscale (bool): Whether to convert the image to grayscale.
    
    Returns:
        np.ndarray: The preprocessed image.
    """
    # Resize the image to the target size
    image_resized = cv2.resize(image, target_size)
    
    # Optionally convert to grayscale if needed by the model.
    if grayscale:
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        image_resized = np.expand_dims(image_resized, axis=-1)
    
    # Normalize pixel values to [0, 1]
    image_normalized = image_resized.astype('float32') / 255.0
    return image_normalized

# ------------------- Image Processing Functions -------------------
def is_blood_report_query(query: str) -> bool:
    """
    Check if the query suggests the user is uploading a blood report.
    """
    keywords = ["blood report", "blood test", "hemoglobin", "CBC", "blood count"]
    return any(keyword in query.lower() for keyword in keywords)

def is_color_image(image: np.ndarray, threshold: float = 10.0) -> bool:
    """
    Determine if an image is in color or effectively grayscale.
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        return False
    b, g, r = cv2.split(image)
    b = b.astype(np.int32)
    g = g.astype(np.int32)
    r = r.astype(np.int32)
    diff_bg = np.mean(np.abs(b - g))
    diff_br = np.mean(np.abs(b - r))
    return not (diff_bg < threshold and diff_br < threshold)

def process_blood_report(image: np.ndarray) -> str:
    """
    Process the blood report image using OCR and return the extracted text.
    """
    ocr_result = pytesseract.image_to_string(image)
    return f"OCR Result: {ocr_result.strip()}"

def process_skin_disease(image: np.ndarray) -> str:
    """
    Process a color image using the skin disease detection model.
    The image is resized to 224x224. This function outputs the confidence scores
    for all disease labels.
    """
    # Resize image to 224×224 for the VGG19-based model
    processed_image = preprocess_image(image, target_size=(224, 224))
    # Predict returns a batch, so extract the first (and only) prediction vector.
    prediction = skin_model.predict(np.expand_dims(processed_image, axis=0))[0]
    
    # Build a result string with all labels and their confidence scores.
    results = []
    for idx, confidence in enumerate(prediction):
        label = skin_label_mapping.get(idx, "Unknown")
        results.append(f"{label}: {confidence:.2f}")
    
    result_str = "Skin Disease Predictions:\n" + "\n".join(results)
    return result_str

def process_xray(image: np.ndarray) -> str:
    """
    Process a grayscale image using the x-ray detection model.
    The image is resized to 256x256, converted to grayscale, and then replicated to create 3 channels.
    This function outputs the confidence scores for both classes.
    """
    processed_image = preprocess_image(image, target_size=(256, 256), grayscale=True)
    # Replicate the grayscale channel to get 3 channels
    processed_image = np.repeat(processed_image, 3, axis=-1)
    prediction = xray_model.predict(np.expand_dims(processed_image, axis=0))[0]
    
    results = []
    for idx, confidence in enumerate(prediction):
        label = xray_label_mapping.get(idx, "Unknown")
        results.append(f"{label}: {confidence:.2f}")
    
    result_str = "X-ray Predictions:\n" + "\n".join(results)
    return result_str

# ------------------- Web Scraping and LLM Integration -------------------
# Google Custom Search API configuration.
google_api_key = "#"
google_cx = "#"  # Replace with your actual Custom Search Engine ID.

def perform_search(query: str, api_key: str, cx: str) -> dict:
    params = {"key": api_key, "cx": cx, "q": query}
    response = requests.get("https://www.googleapis.com/customsearch/v1", params=params, verify=False)
    response.raise_for_status()
    return response.json()

# Azure LLM configuration.
endpoint_url = "#"
api_key_azure = "#"  # Replace with your actual Azure API key
os.environ['AZURE_LLM_DEPLOYMENT_NAME'] = "#"  # Replace with your deployment name.

def query_llm_with_search_results(combined_query: str, search_results: dict) -> str:
    if 'items' in search_results:
        search_results_text = "\n".join(
            f"Snippet: {item.get('snippet', '')}\nLink: {item.get('link', '')}"
            for item in search_results['items']
        )
    else:
        search_results_text = ""
    final_input = f"{combined_query}\n\nAdditional Search Results:\n{search_results_text}"
    
    # Optionally, save for inspection.
    with open("combined_input.txt", "w", encoding="utf-8") as file:
        file.write(final_input)
    print("Combined input has been saved to 'combined_input.txt'.")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key_azure.strip()}"
    }
    data = {
        "messages": [
            {
                "role": "system",
                "content": "You're an expert AI health assistant. When providing an answer, always include the source links received in the input, dont mention about practo or any other medical apps and also dont warn about anything, Refer to only one disease."
            },
            {"role": "user", "content": final_input}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    response = requests.post(endpoint_url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    return generated_text

# ------------------- Main Route -------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    try:
        if request.method == "POST":
            query_text = request.form.get("query_text", "").strip()
            user_location = request.form.get("location", "").strip()
            image_file = request.files.get("image")
            
            model_output = ""
            if image_file and image_file.filename:
                file_bytes = np.frombuffer(image_file.read(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is None:
                    error = "Invalid image file."
                else:
                    if is_blood_report_query(query_text):
                        model_output = process_blood_report(image)
                    else:
                        if is_color_image(image):
                            model_output = process_skin_disease(image)
                        else:
                            model_output = process_xray(image)
            else:
                model_output = "No image provided."
            
            combined_query = (
                f"User Query: {query_text}\n"
                f"Model Output: {model_output}\n"
                f"User Location: {user_location}\n\n"
                "Based on the above details, please provide information on nearby doctors, recommended medicines, "
                "home remedies, and Ayurvedic treatments."
            )
            search_results = perform_search(combined_query, google_api_key, google_cx)
            result = query_llm_with_search_results(combined_query, search_results)
    except Exception as ex:
        error = f"An error occurred: {ex}"
    
    return render_template("index.html", result=result, error=error)

if __name__ == '__main__':
    app.run(debug=True)