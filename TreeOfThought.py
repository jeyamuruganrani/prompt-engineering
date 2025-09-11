import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import GenerationConfig

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

# helper to extract text safely
def extract_text(candidate):
    if candidate.content and candidate.content.parts:
        return "".join([p.text for p in candidate.content.parts if hasattr(p, "text")])
    return ""

def classify_text_tree_of_thoughts(text, num_candidates=3):
    """
    Classify the sentiment of a review using Tree of Thoughts.
    Explores multiple reasoning paths at each stage.
    """
    # Step 1: Generate multiple keyword extractions (thoughts)
    prompt1 = f"""
    Extract all phrases or words in this review that indicate sentiment (positive or negative):

    Review: "{text}"
    Keywords:
    """
    generation_config = GenerationConfig(candidate_count=num_candidates)
    response1 = model.generate_content(prompt1, generation_config=generation_config)
    keyword_options = [extract_text(candidate).strip() for candidate in response1.candidates]

    best_sentiment_path = ""
    best_score = -1

    # Step 2 & 3: For each keyword candidate, classify and aggregate sentiment
    for keywords in keyword_options:
        # Classify keywords
        prompt2 = f"""
        Analyze the following keywords/phrases: {keywords}
        For each, decide if it shows positive or negative sentiment.
        """
        response2 = model.generate_content(prompt2, generation_config=generation_config)
        keyword_sentiments_options = [extract_text(c).strip() for c in response2.candidates]

        # Aggregate sentiment for each reasoning path
        for keyword_sentiments in keyword_sentiments_options:
            prompt3 = f"""
            Based on the analysis of keywords: {keyword_sentiments}, 
            decide the overall sentiment of the review (Positive or Negative) and explain briefly.
            """
            response3 = model.generate_content(prompt3, generation_config=generation_config)

            # Pick the "best" final thought (heuristic: longest explanation)
            for final_thought in response3.candidates:
                thought_text = extract_text(final_thought).strip()
                if len(thought_text) > best_score:
                    best_score = len(thought_text)
                    best_sentiment_path = thought_text

    return best_sentiment_path

# Tree of Thoughts Chatbot loop
print("🤖 Google Gemini Tree of Thoughts Sentiment Classifier (type 'exit' to quit)")
while True:
    user_input = input("Enter a review: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Goodbye! 👋")
        break

    result = classify_text_tree_of_thoughts(user_input)
    print("\nResult:\n", result)
