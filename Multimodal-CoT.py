import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Multimodal model
model = genai.GenerativeModel("gemini-1.5-flash")


# ------------------ Multimodal CoT Classifier ------------------
def classify_text_multimodal_cot(text, stars=None, emoji=None, image_path=None):
    # Step 1: Build multimodal context
    extra_signals = ""
    if stars:
        extra_signals += f"Star Rating: {stars}/5\n"
    if emoji:
        extra_signals += f"Emoji in review: {emoji}\n"

    # Step 2: If image is given → load into Gemini
    inputs = [f"""
    You are a multimodal sentiment classifier. 
    Use step-by-step reasoning that combines:
    - The text review
    - Any extra signals (stars, emoji, image)

    Review: "{text}"
    Extra Signals:
    {extra_signals}

    Reasoning Template:
    1. Extract key positive/negative phrases from the review.
    2. Interpret signals (e.g., stars, emojis, image sentiment cues).
    3. Combine reasoning (text + non-text).
    4. Final Label: Positive or Negative.
    """]

    if image_path:
        with open(image_path, "rb") as img_file:
            inputs.append(img_file.read())

    # Step 3: Call Gemini
    response = model.generate_content(inputs)
    return response.text.strip()


# ------------------ Chatbot Loop ------------------
print("🤖 Google Gemini Multimodal-CoT Classifier (type 'exit' to quit)")
while True:
    user_input = input("Enter a review: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Goodbye! 👋")
        break

    # Example: also pass stars/emoji (simulate multimodal signals)
    result = classify_text_multimodal_cot(user_input, stars=2, emoji="😡")
    print("Result:\n", result)
