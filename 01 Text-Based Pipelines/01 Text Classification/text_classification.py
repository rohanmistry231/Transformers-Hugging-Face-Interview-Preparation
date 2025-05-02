# %% [1. Introduction to Text Classification]
# Learn sentiment analysis and topic classification with Hugging Face pipelines.

# Setup: pip install transformers torch numpy matplotlib
import matplotlib.pyplot as plt
from collections import Counter
from transformers import pipeline

def run_text_classification_demo():
    # %% [2. Synthetic Retail Text Data]
    reviews = [
        "This laptop from TechCorp is great! I love the fast processor.",
        "The screen is vibrant but the battery life is terrible.",
        "Overall, a solid purchase from TechCorp. Highly recommend!"
    ]
    print("Synthetic Text: Retail product reviews created")
    print(f"Reviews: {reviews}")

    # %% [3. Sentiment Analysis]
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_results = classifier(reviews)
    print("Sentiment Analysis: Predictions made")
    for i, (review, result) in enumerate(zip(reviews, sentiment_results)):
        print(f"Review {i+1}: {result['label']} (Score: {result['score']:.2f})")

    # %% [4. Visualization]
    labels = [result['label'] for result in sentiment_results]
    scores = [result['score'] for result in sentiment_results]
    label_counts = Counter(labels)
    plt.figure(figsize=(8, 4))
    plt.bar(label_counts.keys(), label_counts.values(), color=['green' if k == 'POSITIVE' else 'red' for k in label_counts.keys()])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.savefig("text_classification_output.png")
    print("Visualization: Sentiment distribution saved as text_classification_output.png")

    # %% [5. Interview Scenario: Text Classification]
    """
    Interview Scenario: Text Classification
    Q: How does the text-classification pipeline work in Hugging Face?
    A: It uses a pre-trained transformer model (e.g., DistilBERT) to predict labels like positive/negative.
    Key: Fine-tuned on datasets like SST-2 for sentiment analysis.
    Example: pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    """

# Execute the demo
if __name__ == "__main__":
    run_text_classification_demo()