import torch
from PIL import Image
import sys
import requests
import json
import pathlib
temp = pathlib.WindowsPath
pathlib.WindowsPath = pathlib.PosixPath
# Load YOLOv5 model
model = torch.hub.load('./yolov5', 'custom', path='best.pt', source='local', force_reload=True)

# Function to perform object detection using YOLOv5
def perform_object_detection(image_path):
    image_path = image_path.strip()
    img = Image.open(image_path).convert("RGB")
    results = model(img)
    return results

# Function to extract the disease with the highest probability
def extract_diseases(yolov5_results):
    if len(yolov5_results.xyxy[0]) > 0:
        sorted_detections = sorted(yolov5_results.xyxy[0], key=lambda x: x[4], reverse=True)
        highest_confidence_disease = model.names[int(sorted_detections[0][5])]
        confidence = float(sorted_detections[0][4]) * 100
        return highest_confidence_disease, confidence
    else:
        return "No disease detected", 0.0

def initialize_context(image_path):
    yolov5_results = perform_object_detection(image_path)
    disease, confidence = extract_diseases(yolov5_results)
    context = f"The detected disease is {disease} with {confidence:.1f}% confidence."
    return context, disease, confidence

def generate_response(context, text_query):
    prompt = f"""You are SkinGPT, a helpful dermatology assistant. Based on the following context, answer the user's question clearly and in a well-structured format.

Use markdown formatting:
- Use **bold** for key terms
- Use bullet points for lists
- Use headings with ### for sections if needed
- Keep paragraphs short and readable

Context: {context}
User question: {text_query}

Answer:"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.1:8b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.5, "num_predict": 400}
        }
    )
    return response.json()["response"]

