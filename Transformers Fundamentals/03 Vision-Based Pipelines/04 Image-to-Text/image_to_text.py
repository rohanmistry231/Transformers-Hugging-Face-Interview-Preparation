# %% [1. Introduction to Image-to-Text]
# Learn caption generation with Hugging Face image-to-text pipeline.

# Setup: pip install transformers torch numpy matplotlib pillow
import matplotlib.pyplot as plt
from transformers import pipeline
import nltk
import numpy as np

def run_image_to_text_demo():
    # %% [2. Synthetic Image Data Simulation]
    # Note: Due to file I/O constraints, we simulate image inputs with metadata
    images = [
        {"description": "Laptop on a desk", "caption": "A laptop on a wooden desk."},
        {"description": "Smartphone in a store", "caption": "A smartphone displayed in a retail store."},
        {"description": "Broken gadget", "caption": "A broken gadget on a table."}
    ]
    print("Synthetic Images: Simulated retail product images created")
    print(f"Images: {images}")

    # %% [3. Image-to-Text]
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    # Simulate captioning by using predefined captions (since actual image processing requires file input)
    captions = [image["caption"] for image in images]
    print("Image-to-Text: Captions simulated")
    for i, caption in enumerate(captions):
        print(f"Image {i+1}: {caption}")

    # %% [4. Visualization]
    lengths = [len(nltk.word_tokenize(caption)) for caption in captions]
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(captions) + 1), lengths, color='purple')
    plt.title("Caption Lengths")
    plt.xlabel("Image")
    plt.ylabel("Word Count")
    plt.savefig("image_to_text_output.png")
    print("Visualization: Caption lengths saved as image_to_text_output.png")

    # %% [5. Interview Scenario: Image-to-Text]
    """
    Interview Scenario: Image-to-Text
    Q: How does the image-to-text pipeline work in Hugging Face?
    A: It uses multimodal models like BLIP or CLIP to generate text descriptions from image features.
    Key: Combines vision and language transformers for captioning.
    Example: pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_image_to_text_demo()