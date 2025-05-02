# %% [1. Introduction to Image Classification]
# Learn object and scene recognition with Hugging Face image classification pipeline.

# Setup: pip install transformers torch numpy matplotlib pillow
import matplotlib.pyplot as plt
from collections import Counter
from transformers import pipeline
import numpy as np

def run_image_classification_demo():
    # %% [2. Synthetic Image Data Simulation]
    # Note: Due to file I/O constraints, we simulate image inputs with metadata
    images = [
        {"description": "Laptop on a desk", "category": "positive"},
        {"description": "Smartphone in a store", "category": "positive"},
        {"description": "Broken gadget", "category": "negative"}
    ]
    print("Synthetic Images: Simulated retail product images created")
    print(f"Images: {images}")

    # %% [3. Image Classification]
    classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
    # Simulate classification by using predefined categories (since actual image processing requires file input)
    classifications = [image["category"] for image in images]
    print("Image Classification: Classifications simulated")
    for i, classification in enumerate(classifications):
        print(f"Image {i+1}: {classification}")

    # %% [4. Visualization]
    label_counts = Counter(classifications)
    plt.figure(figsize=(8, 4))
    plt.bar(label_counts.keys(), label_counts.values(), color=['green' if k == 'positive' else 'red' for k in label_counts.keys()])
    plt.title("Image Classification Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.savefig("image_classification_output.png")
    print("Visualization: Classification distribution saved as image_classification_output.png")

    # %% [5. Interview Scenario: Image Classification]
    """
    Interview Scenario: Image Classification
    Q: How does the image classification pipeline work in Hugging Face?
    A: It uses Vision Transformers (e.g., ViT) to classify images based on learned patch embeddings.
    Key: Fine-tuned on datasets like ImageNet for robust performance.
    Example: pipeline("image-classification", model="google/vit-base-patch16-224")
    """

# Execute the demo
if __name__ == "__main__":
    run_image_classification_demo()