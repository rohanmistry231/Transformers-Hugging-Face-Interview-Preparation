# %% [1. Introduction to Object Detection]
# Learn bounding box detection with Hugging Face object detection pipeline.

# Setup: pip install transformers torch numpy matplotlib pillow
import matplotlib.pyplot as plt
from collections import Counter
from transformers import pipeline
import numpy as np

def run_object_detection_demo():
    # %% [2. Synthetic Image Data Simulation]
    # Note: Due to file I/O constraints, we simulate image inputs with metadata
    images = [
        {"description": "Laptop and phone on a desk", "objects": ["laptop", "phone"]},
        {"description": "Store shelf with gadgets", "objects": ["phone", "tablet"]},
        {"description": "Broken laptop", "objects": ["laptop"]}
    ]
    print("Synthetic Images: Simulated retail product images created")
    print(f"Images: {images}")

    # %% [3. Object Detection]
    detector = pipeline("object-detection", model="facebook/detr-resnet-50")
    # Simulate detection by using predefined objects (since actual image processing requires file input)
    detections = [image["objects"] for image in images]
    print("Object Detection: Objects simulated")
    for i, objects in enumerate(detections):
        print(f"Image {i+1}: {objects}")

    # %% [4. Visualization]
    all_objects = [obj for detection in detections for obj in detection]
    object_counts = Counter(all_objects)
    plt.figure(figsize=(8, 4))
    plt.bar(object_counts.keys(), object_counts.values(), color='blue')
    plt.title("Detected Object Distribution")
    plt.xlabel("Object")
    plt.ylabel("Count")
    plt.savefig("object_detection_output.png")
    print("Visualization: Object distribution saved as object_detection_output.png")

    # %% [5. Interview Scenario: Object Detection]
    """
    Interview Scenario: Object Detection
    Q: How does the object detection pipeline work in Hugging Face?
    A: It uses models like DETR to predict bounding boxes and class labels for objects in images.
    Key: Combines transformer-based feature extraction with object localization.
    Example: pipeline("object-detection", model="facebook/detr-resnet-50")
    """

# Execute the demo
if __name__ == "__main__":
    run_object_detection_demo()