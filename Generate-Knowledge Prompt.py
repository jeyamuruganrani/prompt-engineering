import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

def classify_text_generate_knowledge(text):
    # Generate-Knowledge Prompt
    prompt = f"""
    You are a helpful text classifier. 

    Step 1: Analyze the review and list all phrases or words that indicate sentiment (positive or negative).  
    Step 2: Based on this analysis, decide whether the review is Positive or Negative.  
    Step 3: Clearly state your reasoning and the final label.

    Review: "{text}"

    Analysis and classification:
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# Generate-Knowledge Chatbot loop
print("🤖 Google Gemini Generate-Knowledge Classifier (type 'exit' to quit)")
while True:
    user_input = input("Enter a review: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Goodbye! 👋")
        break

    result = classify_text_generate_knowledge(user_input)
    print("Result:\n", result)
