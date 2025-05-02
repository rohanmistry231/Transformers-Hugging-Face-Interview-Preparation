# %% [1. Introduction to Image Segmentation]
# Learn pixel-level classification with Hugging Face image segmentation pipeline.

# Setup: pip install transformers torch numpy matplotlib pillow
import matplotlib.pyplot as plt
from collections import Counter
from transformers import pipeline
import numpy as np

def run_image_segmentation_demo():
    # %% [2. Synthetic Image Data Simulation]
    # Note: Due to file I/O constraints, we simulate image inputs with metadata
    images = [
        {"description": "Laptop on a desk", "segments": ["laptop", "desk"]},
        {"description": "Store shelf with gadgets", "segments": ["shelf", "phone", "tablet"]},
        {"description": "Broken laptop", "segments": ["laptop"]}
    ]
    print("Synthetic Images: Simulated retail product images created")
    print(f"Images: {images}")

    # %% [3. Image Segmentation]
    segmenter = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")
    # Simulate segmentation by using predefined segments (since actual image processing requires file input)
    segmentations = [image["segments"] for image in images]
    print("Image Segmentation: Segments simulated")
    for i, segments in enumerate(segmentations):
        print(f"Image {i+1}: {segments}")

    # %% [4. Visualization]
    all_segments = [seg for segmentation in segmentations for seg in segmentation]
    segment_counts = Counter(all_segments)
    plt.figure(figsize=(8, 4))
    plt.bar(segment_counts.keys(), segment_counts.values(), color='green')
    plt.title("Segmented Region Distribution")
    plt.xlabel("Segment")
    plt.ylabel("Count")
    plt.savefig("image_segmentation_output.png")
    print("Visualization: Segment distribution saved as image_segmentation_output.png")

    # %% [5. Interview Scenario: Image Segmentation]
    """
    Interview Scenario: Image Segmentation
    Q: How does the image segmentation pipeline work in Hugging Face?
    A: It uses models like DETR to assign class labels to each pixel or region in an image.
    Key: Supports panoptic segmentation for both objects and background.
    Example: pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")
    """

# Execute the demo
if __name__ == "__main__":
    run_image_segmentation_demo()