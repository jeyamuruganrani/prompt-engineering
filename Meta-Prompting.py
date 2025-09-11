import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")  # or "gemini-1.5-pro"

def classify_text(text):
    # Meta-Prompting
    prompt = f"""
    You are a helpful text classifier. Follow these instructions:

    1. Read the review carefully.
    2. Think step by step about whether the review expresses a positive or negative opinion.
    3. Explain your reasoning in simple words.
    4. At the end, clearly state the label as either Positive or Negative.

    Review: "{text}"

    Step-by-step reasoning and final label:
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# Meta-Prompting Chatbot loop
print("🤖 Google Gemini Meta-Prompting Classifier (type 'exit' to quit)")
while True:
    user_input = input("Enter a review: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Goodbye! 👋")
        break

    result = classify_text(user_input)
    print("Result:\n", result)
