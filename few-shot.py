import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env
load_dotenv()

# Configure API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Create model
model = genai.GenerativeModel("gemini-1.5-flash")  # or "gemini-1.5-pro"


def classify_text(text):
    # Few-shot examples
    prompt = f"""
    Classify each review as Positive or Negative.

    Example 1:
    Review: "I really enjoyed the movie, it was amazing!"
    Label: Positive

    Example 2:
    Review: "The food was terrible and I will not come back."
    Label: Negative

    Example 3:
    Review: "It was okay, but not great."
    Label: Negative

    Now classify this review:
    Review: "{text}"
    Label:
    """

    response = model.generate_content(prompt)
    return response.text.strip()


# Few-Shot Chatbot loop
print("🤖 Google Gemini Few-Shot Classifier (type 'exit' to quit)")
while True:
    user_input = input("Enter a review: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Goodbye! 👋")
        break

    result = classify_text(user_input)
    print("Classification:", result)
