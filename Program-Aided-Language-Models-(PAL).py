import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

# ------------------ PAL Classifier ------------------
def classify_text_pal(text):
    # Step 1: Ask LLM to write Python code for analysis
    code_prompt = f"""
    You are a Program-Aided Language Model (PAL).
    Task: Write a Python function `classify_sentiment(review)` that:
    - Splits review into words
    - Counts positive and negative words
    - Returns a dictionary with counts and final label ("Positive" or "Negative")

    Review: "{text}"

    Just output the code only (no explanation).
    """
    code_response = model.generate_content(code_prompt).text

    # Step 2: Execute the code safely
    local_vars = {}
    try:
        exec(code_response, {}, local_vars)
        classify_func = local_vars.get("classify_sentiment")
        tool_result = classify_func(text)
    except Exception as e:
        tool_result = {"error": str(e)}

    # Step 3: Ask model to combine reasoning + tool output
    reasoning_prompt = f"""
    Review: "{text}"
    Tool Output: {tool_result}

    Step 1: Explain key sentiment words from the tool output.
    Step 2: Confirm the sentiment label.
    Step 3: Return final classification.
    """
    response = model.generate_content(reasoning_prompt)
    return response.text.strip()

# ------------------ Chatbot Loop ------------------
print("🤖 Google Gemini PAL Classifier (type 'exit' to quit)")
while True:
    user_input = input("Enter a review: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Goodbye! 👋")
        break

    result = classify_text_pal(user_input)
    print("Result:\n", result)
