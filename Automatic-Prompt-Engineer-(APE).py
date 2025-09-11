import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")


# ------------------ Step 1: Generate Candidate Prompts ------------------
def generate_prompts(task_description):
    prompt = f"""
    You are an Automatic Prompt Engineer (APE).

    Task: {task_description}

    Step 1: Generate 3 different candidate prompts for this task.
    Step 2: Each prompt should guide the AI clearly with step-by-step instructions.
    Step 3: Return them as a numbered list.
    """
    response = model.generate_content(prompt)
    return response.text.strip()


# ------------------ Step 2: Select Best Prompt ------------------
def select_best_prompt(candidates, review):
    prompt = f"""
    You are an evaluator. Below are 3 candidate prompts for sentiment classification.

    Candidate Prompts:
    {candidates}

    Task: Choose the best prompt that will give the most accurate and well-reasoned classification
    for the following review: "{review}".

    Return only the chosen prompt text.
    """
    response = model.generate_content(prompt)
    return response.text.strip()


# ------------------ Step 3: Use Best Prompt for Classification ------------------
def classify_text_ape(text):
    task = "Classify a review as Positive or Negative with explanation."

    # Generate candidate prompts
    candidates = generate_prompts(task)

    # Select the best prompt for this review
    best_prompt = select_best_prompt(candidates, text)

    # Final run with best prompt
    final_prompt = f"""
    {best_prompt}

    Review: "{text}"
    """
    response = model.generate_content(final_prompt)
    return response.text.strip()


# ------------------ Chatbot Loop ------------------
print("🤖 Google Gemini APE Classifier (type 'exit' to quit)")
while True:
    user_input = input("Enter a review: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Goodbye! 👋")
        break

    result = classify_text_ape(user_input)
    print("Result:\n", result)
