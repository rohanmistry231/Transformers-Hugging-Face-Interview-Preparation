# %% [1. Introduction to Fill-Mask]
# Learn masked language modeling with Hugging Face fill-mask pipeline.

# Setup: pip install transformers torch numpy matplotlib
import matplotlib.pyplot as plt
from transformers import pipeline

def run_fill_mask_demo():
    # %% [2. Synthetic Retail Text Data]
    masked_texts = [
        "This laptop from TechCorp is [MASK]!",
        "The [MASK] is vibrant but the battery life is terrible.",
        "Overall, a [MASK] purchase from TechCorp."
    ]
    print("Synthetic Text: Retail masked texts created")
    print(f"Masked Texts: {masked_texts}")

    # %% [3. Masked Language Modeling]
    fill_mask = pipeline("fill-mask", model="bert-base-uncased")
    predictions = [fill_mask(text)[:3] for text in masked_texts]  # Top 3 predictions
    print("Fill-Mask: Predictions made")
    for i, (text, preds) in enumerate(zip(masked_texts, predictions)):
        print(f"Text {i+1}: {text}")
        for j, pred in enumerate(preds):
            print(f"Prediction {j+1}: {pred['token_str']} (Score: {pred['score']:.2f})")

    # %% [4. Visualization]
    scores = [[pred['score'] for pred in preds] for preds in predictions]
    plt.figure(figsize=(8, 4))
    for i, score_list in enumerate(scores):
        plt.bar([x + i*0.3 for x in range(1, len(score_list) + 1)], score_list, width=0.3, label=f"Text {i+1}")
    plt.title("Prediction Confidence Scores")
    plt.xlabel("Prediction Rank")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig("fill_mask_output.png")
    print("Visualization: Prediction confidence saved as fill_mask_output.png")

    # %% [5. Interview Scenario: Fill-Mask]
    """
    Interview Scenario: Fill-Mask
    Q: How does the fill-mask pipeline leverage masked language models?
    A: It uses models like BERT to predict masked tokens based on context.
    Key: Trained on large corpora to understand word relationships.
    Example: pipeline("fill-mask", model="bert-base-uncased")
    """

# Execute the demo
if __name__ == "__main__":
    run_fill_mask_demo()