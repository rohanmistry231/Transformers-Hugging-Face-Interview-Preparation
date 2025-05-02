# %% [1. Introduction to Translation]
# Learn multilingual translation with Hugging Face pipelines.

# Setup: pip install transformers torch numpy matplotlib
import matplotlib.pyplot as plt
from transformers import pipeline

def run_translation_demo():
    # %% [2. Synthetic Retail Text Data]
    reviews = [
        "This laptop from TechCorp is great!",
        "The screen is vibrant but the battery life is terrible.",
        "Overall, a solid purchase from TechCorp."
    ]
    target_languages = ["es", "fr"]  # Spanish, French
    print("Synthetic Text: Retail product reviews created")
    print(f"Reviews: {reviews}")

    # %% [3. Multilingual Translation]
    translations = []
    for lang in target_languages:
        translator = pipeline(f"translation_en_to_{lang}", model=f"Helsinki-NLP/opus-mt-en-{lang}")
        lang_translations = [translator(review)[0]['translation_text'] for review in reviews]
        translations.append((lang, lang_translations))
    print("Translation: Texts translated")
    for lang, trans in translations:
        print(f"Language: {lang.upper()}")
        for i, t in enumerate(trans):
            print(f"Review {i+1}: {t}")

    # %% [4. Visualization]
    lengths = [[len(t.split()) for t in trans] for lang, trans in translations]
    plt.figure(figsize=(8, 4))
    for i, (lang, lens) in enumerate(zip(target_languages, lengths)):
        plt.bar([x + i*0.4 for x in range(1, len(reviews) + 1)], lens, width=0.4, label=lang.upper())
    plt.title("Translation Lengths by Language")
    plt.xlabel("Review")
    plt.ylabel("Word Count")
    plt.legend()
    plt.savefig("translation_output.png")
    print("Visualization: Translation lengths saved as translation_output.png")

    # %% [5. Interview Scenario: Translation]
    """
    Interview Scenario: Translation
    Q: How does the translation pipeline work in Hugging Face?
    A: It uses encoder-decoder models (e.g., MarianMT) fine-tuned for language pairs.
    Key: Supports multilingual translation with high accuracy.
    Example: pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
    """

# Execute the demo
if __name__ == "__main__":
    run_translation_demo()