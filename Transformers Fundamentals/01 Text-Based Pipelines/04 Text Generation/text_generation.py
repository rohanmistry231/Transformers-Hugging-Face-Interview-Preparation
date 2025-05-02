# %% [1. Introduction to Text Generation]
# Learn story generation and text completion with Hugging Face pipelines.

# Setup: pip install transformers torch numpy matplotlib
import matplotlib.pyplot as plt
from transformers import pipeline
import nltk

def run_text_generation_demo():
    # %% [2. Synthetic Retail Text Data]
    prompts = [
        "The new TechCorp laptop is amazing because",
        "A customer review of the vibrant screen:",
        "Why I love shopping at TechCorp:"
    ]
    print("Synthetic Text: Retail text prompts created")
    print(f"Prompts: {prompts}")

    # %% [3. Text Generation]
    generator = pipeline("text-generation", model="gpt2", max_length=50)
    generated_texts = [generator(prompt, num_return_sequences=1)[0]['generated_text'] for prompt in prompts]
    print("Text Generation: Texts generated")
    for i, (prompt, text) in enumerate(zip(prompts, generated_texts)):
        print(f"Prompt {i+1}: {prompt}")
        print(f"Generated: {text[:100]}...")

    # %% [4. Visualization]
    lengths = [len(nltk.word_tokenize(text)) for text in generated_texts]
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(prompts) + 1), lengths, color='purple')
    plt.title("Generated Text Lengths")
    plt.xlabel("Prompt")
    plt.ylabel("Word Count")
    plt.savefig("text_generation_output.png")
    print("Visualization: Generated text lengths saved as text_generation_output.png")

    # %% [5. Interview Scenario: Text Generation]
    """
    Interview Scenario: Text Generation
    Q: How does the text-generation pipeline work in Hugging Face?
    A: It uses a generative model (e.g., GPT-2) to predict the next token iteratively.
    Key: Controlled by parameters like max_length and num_return_sequences.
    Example: pipeline("text-generation", model="gpt2")
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_text_generation_demo()