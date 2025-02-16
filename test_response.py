import os
import json
from google import genai
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
client = genai.Client(api_key=api_key)

# Hardcoded Title and Abstract for testing Gemini
title = "3D Copy-Paste: Physically Plausible Object Insertion"
abstract = """A major challenge in monocular 3D object detection is the limited diversity and quantity of objects in real datasets. While augmenting real scenes with virtual objects 
holds promise to improve both the diversity and quantity of the objects, it remains elusive due to the lack of an effective 3D object insertion method in complex real captured 
scenes. In this work, we study augmenting complex real indoor scenes with virtual objects for monocular 3D object detection. The main challenge is to automatically identify 
plausible physical properties for virtual assets (e.g., locations, appearances, sizes, etc.) in cluttered real scenes. To address this challenge, we propose a physically plausible 
indoor 3D object insertion approach to automatically copy virtual objects and paste them into real scenes. The resulting objects in scenes have 3D bounding boxes with plausible 
physical locations and appearances. In particular, our method first identifies physically feasible locations and poses for the inserted objects to prevent collisions with the 
existing room layout. Subsequently, it estimates spatially-varying illumination for the insertion location, enabling the immersive blending of the virtual objects into the original 
scene with plausible appearances and cast shadows. We show that our augmentation method significantly improves existing monocular 3D object models and achieves state-of-the-art 
performance. For the first time, we demonstrate that a physically plausible 3D object insertion, serving as a generative data augmentation technique, can lead to significant 
improvements for discriminative downstream tasks such as monocular 3D object detection. Project website: https://gyhandy.github.io/3D-Copy-Paste/."""

# Function to classify research paper and provide a reason
def classify_paper(title, abstract):
    prompt = f"""
    Classify the following research paper into one of these categories:
    - Deep Learning
    - Computer Vision
    - Reinforcement Learning
    - NLP
    - Optimization

    Title: {title}
    Abstract: {abstract}

    Return the output in this exact format:
    Category: [Predicted Category]
    Reason: [One-line explanation]
    """

    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)

    # Extract category and reason
    response_text = response.text.strip().split("\n")
    category = response_text[0].replace("Category:", "").strip() if len(response_text) > 0 else "Unknown"
    reason = response_text[1].replace("Reason:", "").strip() if len(response_text) > 1 else "No explanation provided."

    return category, reason

# Testing Gemini Classification
category, reason = classify_paper(title, abstract)

# Display Results
print("\n--- Gemini Classification Result ---")
print(f"Category: {category}")
print(f"Reason: {reason}\n")

# Save to JSON
json_data = {
    "title": title,
    "category": category,
    "reason": reason
}

with open("gemini_test_result.json", "w") as f:
    json.dump(json_data, f, indent=4)

print("Results saved to 'gemini_test_result.json'.")
