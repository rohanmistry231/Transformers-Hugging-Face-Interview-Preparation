# %% [1. Introduction to Named Entity Recognition]
# Learn entity extraction with Hugging Face NER pipeline.

# Setup: pip install transformers torch numpy matplotlib
import matplotlib.pyplot as plt
from collections import Counter
from transformers import pipeline

def run_ner_demo():
    # %% [2. Synthetic Retail Text Data]
    reviews = [
        "This laptop from TechCorp is great! I love the fast processor from Intel.",
        "The screen is vibrant, designed by Samsung in New York.",
        "Overall, a solid purchase from TechCorp in California."
    ]
    print("Synthetic Text: Retail product reviews created")
    print(f"Reviews: {reviews}")

    # %% [3. Entity Extraction]
    ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
    entities = []
    for review in reviews:
        result = ner(review)
        entities.extend([(entity['entity_group'], entity['word']) for entity in result])
    print("NER: Entities extracted")
    print(f"Entities (Sample): {entities[:5]}...")

    # %% [4. Visualization]
    entity_types = [entity[0] for entity in entities]
    type_counts = Counter(entity_types)
    plt.figure(figsize=(8, 4))
    plt.bar(type_counts.keys(), type_counts.values(), color='blue')
    plt.title("Entity Type Distribution")
    plt.xlabel("Entity Type")
    plt.ylabel("Count")
    plt.savefig("ner_output.png")
    print("Visualization: Entity distribution saved as ner_output.png")

    # %% [5. Interview Scenario: NER]
    """
    Interview Scenario: Named Entity Recognition
    Q: How does the NER pipeline identify entities in Hugging Face?
    A: It uses a transformer model (e.g., BERT) fine-tuned to classify tokens into entity categories.
    Key: Groups tokens into entities like PERSON, ORG, LOC.
    Example: pipeline("ner", model="dslim/bert-base-NER")
    """

# Execute the demo
if __name__ == "__main__":
    run_ner_demo()