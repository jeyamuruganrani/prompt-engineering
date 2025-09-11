import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")  # or "gemini-1.5-pro"


def classify_text(text):
    # Chain-of-Thought prompt
    prompt = f"""
    Classify this review as Positive or Negative. 
    Think step by step and explain your reasoning before giving the final label.

    Review: "{text}"

    Step-by-step reasoning:
    """
    response = model.generate_content(prompt)
    return response.text.strip()


# CoT Chatbot loop
print("🤖 Google Gemini CoT Classifier (type 'exit' to quit)")
while True:
    user_input = input("Enter a review: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Goodbye! 👋")
        break

    result = classify_text(user_input)
    print("Result:\n", result)
