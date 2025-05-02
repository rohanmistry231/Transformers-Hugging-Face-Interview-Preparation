# %% [1. Introduction to Summarization]
# Learn abstractive and extractive summarization with Hugging Face pipelines.

# Setup: pip install transformers torch numpy matplotlib
import matplotlib.pyplot as plt
from transformers import pipeline
import nltk

def run_summarization_demo():
    # %% [2. Synthetic Retail Text Data]
    texts = [
        """
        TechCorp's new laptop has a fast processor from Intel and a vibrant screen designed by Samsung.
        The battery life is average, lasting about 6 hours. It was launched in New York in 2025.
        Customers love the sleek design and performance but some complain about the battery.
        """
    ]
    print("Synthetic Text: Retail product description created")
    print(f"Text: {texts[0][:100]}...")

    # %% [3. Abstractive Summarization]
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = [summarizer(text, max_length=50, min_length=10, do_sample=False)[0]['summary_text'] for text in texts]
    print("Summarization: Summaries generated")
    for i, summary in enumerate(summaries):
        print(f"Summary {i+1}: {summary}")

    # %% [4. Visualization]
    lengths = [len(nltk.word_tokenize(summary)) for summary in summaries]
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(summaries) + 1), lengths, color='orange')
    plt.title("Summary Lengths")
    plt.xlabel("Summary")
    plt.ylabel("Word Count")
    plt.savefig("summarization_output.png")
    print("Visualization: Summary lengths saved as summarization_output.png")

    # %% [5. Interview Scenario: Summarization]
    """
    Interview Scenario: Summarization
    Q: What’s the difference between abstractive and extractive summarization?
    A: Abstractive generates new text; extractive selects existing sentences.
    Key: Abstractive uses models like BART, extractive uses algorithms like TextRank.
    Example: pipeline("summarization", model="facebook/bart-large-cnn")
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_summarization_demo()