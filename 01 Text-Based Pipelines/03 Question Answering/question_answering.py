# %% [1. Introduction to Question Answering]
# Learn extractive and generative QA with Hugging Face pipelines.

# Setup: pip install transformers torch numpy matplotlib
import matplotlib.pyplot as plt
from transformers import pipeline

def run_question_answering_demo():
    # %% [2. Synthetic Retail Text Data]
    context = """
    TechCorp's new laptop has a fast processor from Intel and a vibrant screen designed by Samsung.
    The battery life is average, lasting about 6 hours. It was launched in New York in 2025.
    """
    questions = [
        "What is the processor brand?",
        "Where was the laptop launched?",
        "How long does the battery last?"
    ]
    print("Synthetic Text: Retail product description created")
    print(f"Context: {context[:100]}...")
    print(f"Questions: {questions}")

    # %% [3. Extractive QA]
    qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    answers = [qa(question=question, context=context) for question in questions]
    print("Question Answering: Answers extracted")
    for i, (question, answer) in enumerate(zip(questions, answers)):
        print(f"Question {i+1}: {question}")
        print(f"Answer: {answer['answer']} (Score: {answer['score']:.2f})")

    # %% [4. Visualization]
    scores = [answer['score'] for answer in answers]
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(questions) + 1), scores, color='green')
    plt.title("Answer Confidence Scores")
    plt.xlabel("Question")
    plt.ylabel("Confidence Score")
    plt.savefig("question_answering_output.png")
    print("Visualization: Answer confidence saved as question_answering_output.png")

    # %% [5. Interview Scenario: Question Answering]
    """
    Interview Scenario: Question Answering
    Q: What’s the difference between extractive and generative QA?
    A: Extractive QA selects spans from the context; generative QA generates free-form answers.
    Key: Extractive uses models like BERT, generative uses T5 or GPT.
    Example: pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    """

# Execute the demo
if __name__ == "__main__":
    run_question_answering_demo()