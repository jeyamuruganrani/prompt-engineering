import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

# ------------------ TOOL: Simple sentiment lexicon ------------------
positive_words = {"amazing", "great", "good", "wonderful", "fantastic", "excellent", "love", "happy"}
negative_words = {"bad", "terrible", "awful", "horrible", "worst", "hate", "disappointed", "sad"}


def sentiment_tool(text):
    tokens = text.lower().split()
    pos = sum(1 for t in tokens if t in positive_words)
    neg = sum(1 for t in tokens if t in negative_words)

    if pos > neg:
        return f"Tool Result: Positive ({pos} positive vs {neg} negative words)"
    elif neg > pos:
        return f"Tool Result: Negative ({neg} negative vs {pos} positive words)"
    else:
        return f"Tool Result: Neutral (equal positives and negatives)"


# ------------------ Automatic Reasoning & Tool-use ------------------
def classify_text_art(text):
    # Step 1: Ask Gemini to reason if tool is needed
    decision_prompt = f"""
    You are an intelligent assistant with access to a sentiment analysis tool.

    Review: "{text}"

    Step 1: Decide if you can classify sentiment directly OR if you should call the tool for extra help.
    Step 2: If tool-use is required, say: "TOOL-USE".
    Step 3: Otherwise, classify directly as Positive or Negative.
    """

    decision = model.generate_content(decision_prompt).text.strip()

    # Step 2: If tool is needed → Call the tool
    if "TOOL-USE" in decision:
        tool_result = sentiment_tool(text)

        reasoning_prompt = f"""
        You are a sentiment classifier. You have access to both the review and the tool output.

        Review: "{text}"
        {tool_result}

        Step 1: Combine your reasoning with the tool output.
        Step 2: Explain the key positive/negative signals.
        Step 3: Give final label: Positive or Negative.
        """

        response = model.generate_content(reasoning_prompt)
        return response.text.strip()

    # Step 3: Otherwise, use model-only classification
    return decision


# ------------------ Chatbot Loop ------------------
print("🤖 Google Gemini Automatic Reasoning + Tool-use Classifier (type 'exit' to quit)")
while True:
    user_input = input("Enter a review: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Goodbye! 👋")
        break

    result = classify_text_art(user_input)
    print("Result:\n", result)
