import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")


# ------------------ Graph Prompting Classifier ------------------
def classify_text_graph_prompt(text):
    prompt = f"""
    You are a sentiment classifier using **Graph Prompting**.

    Task:
    1. Build a sentiment reasoning graph from the review.
       - Nodes: key phrases, words, or metadata (positive or negative).
       - Edges: how these phrases connect to final sentiment.
    2. Use the graph to decide sentiment.
    3. Provide reasoning and the final label.

    Example format:

    Review: "The product is amazing but delivery was late."

    Graph:
      [Product is amazing] --positive--> [Sentiment]
      [Delivery was late] --negative--> [Sentiment]

    Reasoning:
      - Positive node stronger than negative node.
    Final Label: Positive

    Now classify this review:

    Review: "{text}"
    """
    response = model.generate_content(prompt)
    return response.text.strip()


# ------------------ Chatbot Loop ------------------
print("🤖 Google Gemini Graph-Prompting Classifier (type 'exit' to quit)")
while True:
    user_input = input("Enter a review: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Goodbye! 👋")
        break

    result = classify_text_graph_prompt(user_input)
    print("Result:\n", result)
