import os
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

# ------------------ Knowledge Base ------------------
knowledge_base = [
    "Positive words: amazing, wonderful, great, excellent, good, fantastic, love, happy",
    "Negative words: bad, terrible, awful, horrible, worst, hate, disappointed, sad",
    "Sentiment classification rule: If more positive words → Positive. If more negative words → Negative."
]

# ------------------ Embedding Helper ------------------
def embed_text(text):
    emb = genai.embed_content(model="models/embedding-001", content=text)
    return np.array(emb["embedding"])

# Precompute embeddings for KB
kb_embeddings = [embed_text(doc) for doc in knowledge_base]

# ------------------ Retrieve ------------------
def retrieve_context(query, top_k=2):
    query_emb = embed_text(query)
    sims = [cosine_similarity([query_emb], [doc_emb])[0][0] for doc_emb in kb_embeddings]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [knowledge_base[i] for i in top_indices]

# ------------------ RAG Classifier ------------------
def classify_text_rag(text):
    context_docs = retrieve_context(text)
    context = "\n".join(context_docs)

    prompt = f"""
    You are a helpful sentiment classifier.

    Context:
    {context}

    Review: "{text}"

    Step 1: Use the above context rules & keywords to analyze sentiment.
    Step 2: Identify key positive/negative words.
    Step 3: Decide if the sentiment is Positive or Negative.
    Step 4: Explain reasoning and give final label.
    """

    response = model.generate_content(prompt)
    return response.text.strip()

# ------------------ Chatbot Loop ------------------
print("🤖 Google Gemini RAG Classifier (type 'exit' to quit)")
while True:
    user_input = input("Enter a review: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Goodbye! 👋")
        break

    result = classify_text_rag(user_input)
    print("Result:\n", result)
