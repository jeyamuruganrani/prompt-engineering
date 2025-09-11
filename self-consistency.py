import os
from dotenv import load_dotenv
import google.generativeai as genai
from collections import Counter

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")


# ------------------ Generate candidate reasoning chains ------------------
def generate_candidates(text, n=5):
    prompt = f"""
    You are a sentiment classifier.

    Review: "{text}"

    Task:
    - Think step by step.
    - Identify key positive/negative words.
    - Explain reasoning.
    - Give Final Label: Positive or Negative.

    Repeat this reasoning {n} times, generating {n} independent candidate answers.
    """
    response = model.generate_content(prompt)
    # Split responses if necessary (Gemini may return numbered list)
    candidates = response.text.strip().split("\n")
    return candidates[:n]


# ------------------ Extract final label ------------------
def extract_label(candidate_text):
    for line in candidate_text.splitlines():
        if "Positive" in line:
            return "Positive"
        if "Negative" in line:
            return "Negative"
    return "Neutral"


# ------------------ Self-Consistency Classifier ------------------
def classify_text_self_consistency(text, n=5):
    candidates = generate_candidates(text, n)
    labels = [extract_label(c) for c in candidates]

    # Majority voting
    most_common = Counter(labels).most_common(1)[0][0]

    return {
        "candidates": candidates,
        "final_label": most_common
    }


# ------------------ Chatbot Loop ------------------
print("🤖 Google Gemini Self-Consistency Classifier (type 'exit' to quit)")
while True:
    user_input = input("Enter a review: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Goodbye! 👋")
        break

    result = classify_text_self_consistency(user_input)
    print("Candidate Reasonings:\n", "\n---\n".join(result["candidates"]))
    print("\nFinal Label (majority vote):", result["final_label"])
