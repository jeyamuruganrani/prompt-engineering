import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

def classify_text_prompt_chaining(text):
    # Step 1: Extract keywords
    prompt1 = f"""
    Extract all phrases or words in this review that indicate sentiment (positive or negative):

    Review: "{text}"
    Keywords:
    """
    response1 = model.generate_content(prompt1)
    keywords = response1.text.strip()

    # Step 2: Classify keywords
    prompt2 = f"""
    Analyze the following keywords/phrases: {keywords}
    For each, decide if it shows positive or negative sentiment.
    """
    response2 = model.generate_content(prompt2)
    keyword_sentiments = response2.text.strip()

    # Step 3: Aggregate and give final label
    prompt3 = f"""
    Based on the analysis of keywords: {keyword_sentiments}, 
    decide the overall sentiment of the review (Positive or Negative) and explain briefly.
    """
    response3 = model.generate_content(prompt3)
    final_result = response3.text.strip()

    return final_result

# Prompt Chaining Chatbot loop
print("🤖 Google Gemini Prompt Chaining Classifier (type 'exit' to quit)")
while True:
    user_input = input("Enter a review: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Goodbye! 👋")
        break

    result = classify_text_prompt_chaining(user_input)
    print("Result:\n", result)
