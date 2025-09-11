import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

# ------------------ Step 1: Generate Candidate Answers ------------------
def generate_candidates(text, num_candidates=3):
    prompt = f"""
    You are a sentiment classifier.

    Review: "{text}"

    Task: Generate {num_candidates} different reasoning paths and sentiment labels.
    Each candidate should:
    - Identify key positive/negative words
    - Explain reasoning
    - End with Final Label: Positive or Negative
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# ------------------ Step 2: Evaluate & Pick Best ------------------
def select_best_answer(candidates, review):
    prompt = f"""
    You are an evaluator. Below are multiple candidate answers for the review: "{review}".

    Candidates:
    {candidates}

    Task:
    - Compare the answers.
    - Select the one with the clearest reasoning and most accurate label.
    - Return only the best answer.
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# ------------------ Step 3: Active-Prompt Classifier ------------------
def classify_text_active_prompt(text):
    candidates = generate_candidates(text)
    best_answer = select_best_answer(candidates, text)
    return best_answer

# ------------------ Chatbot Loop ------------------
print("🤖 Google Gemini Active-Prompt Classifier (type 'exit' to quit)")
while True:
    user_input = input("Enter a review: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Goodbye! 👋")
        break

    result = classify_text_active_prompt(user_input)
    print("Result:\n", result)
