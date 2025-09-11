import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

# ------------------ TOOL ------------------
positive_words = {"amazing", "great", "good", "wonderful", "fantastic", "excellent", "love", "happy"}
negative_words = {"bad", "terrible", "awful", "horrible", "worst", "hate", "disappointed", "sad"}


def sentiment_tool(text):
    tokens = text.lower().split()
    pos = sum(1 for t in tokens if t in positive_words)
    neg = sum(1 for t in tokens if t in negative_words)

    if pos > neg:
        return {"label": "Positive", "pos": pos, "neg": neg}
    elif neg > pos:
        return {"label": "Negative", "pos": pos, "neg": neg}
    else:
        return {"label": "Neutral", "pos": pos, "neg": neg}


# ------------------ ReAct Classifier ------------------
def classify_text_react(text):
    # Step 1: Model decides whether to use reasoning or tool
    decision_prompt = f"""
    You are a ReAct agent (Reasoning + Acting).

    Review: "{text}"

    Think step by step:
    - First, reason about whether this text clearly shows positive or negative sentiment.
    - If it's not clear enough, decide to ACT and call the sentiment tool.
    - If tool is used, include its output as OBSERVATION in your reasoning.
    - End with Final Answer: Positive or Negative.
    """

    # Step 2: Get model reasoning (simulate decision)
    reasoning = model.generate_content(decision_prompt).text.strip()

    # Step 3: If model wants to ACT → Call tool
    if "ACT" in reasoning or "TOOL" in reasoning:
        tool_result = sentiment_tool(text)

        reasoning_with_obs = f"""
        Review: "{text}"
        Observation from Tool: {tool_result}

        Continue reasoning using the observation.
        Give final classification (Positive or Negative) with explanation.
        """
        response = model.generate_content(reasoning_with_obs)
        return response.text.strip()

    # Step 4: Otherwise → Direct answer
    return reasoning


# ------------------ Chatbot Loop ------------------
print("🤖 Google Gemini ReAct Classifier (type 'exit' to quit)")
while True:
    user_input = input("Enter a review: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Goodbye! 👋")
        break

    result = classify_text_react(user_input)
    print("Result:\n", result)
