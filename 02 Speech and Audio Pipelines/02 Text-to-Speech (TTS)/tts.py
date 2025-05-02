# %% [1. Introduction to Text-to-Speech]
# Learn speech synthesis with Hugging Face TTS pipeline.

# Setup: pip install transformers torch numpy matplotlib
import matplotlib.pyplot as plt
from transformers import pipeline
import numpy as np

def run_tts_demo():
    # %% [2. Synthetic Retail Text Data]
    texts = [
        "Welcome to TechCorp! Our new laptop is amazing.",
        "The vibrant screen is a customer favorite.",
        "Visit our store for exclusive deals today."
    ]
    print("Synthetic Text: Retail announcements created")
    print(f"Texts: {texts}")

    # %% [3. TTS Pipeline Simulation]
    # Note: TTS pipeline generates audio; we simulate metadata due to file output constraints
    tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")
    # Simulate TTS output with estimated durations (seconds per word approximation)
    durations = [len(text.split()) * 0.5 for text in texts]  # Approx 0.5s per word
    print("TTS: Audio generation simulated")
    for i, (text, duration) in enumerate(zip(texts, durations)):
        print(f"Text {i+1}: {text}")
        print(f"Simulated Duration: {duration:.2f} seconds")

    # %% [4. Visualization]
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(texts) + 1), durations, color='green')
    plt.title("Simulated Audio Durations")
    plt.xlabel("Text Sample")
    plt.ylabel("Duration (Seconds)")
    plt.savefig("tts_output.png")
    print("Visualization: Audio durations saved as tts_output.png")

    # %% [5. Interview Scenario: TTS]
    """
    Interview Scenario: Text-to-Speech
    Q: How does the TTS pipeline synthesize speech in Hugging Face?
    A: It uses models like SpeechT5 or MMS-TTS to generate audio waveforms from text embeddings.
    Key: Trained on speech datasets to produce natural-sounding audio.
    Example: pipeline("text-to-speech", model="facebook/mms-tts-eng")
    """

# Execute the demo
if __name__ == "__main__":
    run_tts_demo()