import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

# ------------------ Directional Stimulus Prompting ------------------
def classify_text_dsp(text):
    prompt = f"""
    You are a sentiment classifier. 
    Use the following *directional stimulus template* to ensure consistent reasoning:

    Template:
    - Step 1: Extract sentiment-related words/phrases from the review.
    - Step 2: Count positive vs negative words.
    - Step 3: Decide sentiment (Positive or Negative).
    - Step 4: Explain reasoning briefly.
    - Step 5: Final Label.

    Review: "{text}"

    Follow the template strictly.
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# ------------------ Chatbot Loop ------------------
print("🤖 Google Gemini DSP Classifier (type 'exit' to quit)")
while True:
    user_input = input("Enter a review: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Goodbye! 👋")
        break

    result = classify_text_dsp(user_input)
    print("Result:\n", result)
