import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")


# ------------------ Reflexion Classifier ------------------
def classify_text_reflexion(text):
    # Step 1: First attempt
    first_prompt = f"""
    You are a sentiment classifier.

    Review: "{text}"

    Step 1: Identify sentiment-related words.
    Step 2: Explain reasoning.
    Step 3: Final Label: Positive or Negative.
    """
    first_response = model.generate_content(first_prompt).text.strip()

    # Step 2: Reflection
    reflection_prompt = f"""
    You previously classified the review: "{text}"

    Your answer was:
    {first_response}

    Now reflect:
    - Was the reasoning logically sound?
    - Did you miss any important sentiment words?
    - Is the final label consistent with the analysis?

    If the answer was correct → confirm it.
    If mistakes are found → correct them and provide an improved answer.
    """
    reflection_response = model.generate_content(reflection_prompt).text.strip()

    return reflection_response


# ------------------ Chatbot Loop ------------------
print("🤖 Google Gemini Reflexion Classifier (type 'exit' to quit)")
while True:
    user_input = input("Enter a review: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Goodbye! 👋")
        break

    result = classify_text_reflexion(user_input)
    print("Result:\n", result)
