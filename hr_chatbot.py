import json
import spacy
import torch
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz

# Load HR FAQs (Updated to expanded dataset)
FAQ_FILE = "expanded_hr_faq.json"
try:
    with open(FAQ_FILE, "r", encoding="utf-8") as file:
        hr_faqs = json.load(file)
except FileNotFoundError:
    print(f"Error: '{FAQ_FILE}' file not found. Make sure it is in the same directory as this script.")
    exit()

# Load SBERT model for better semantic search
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract questions and their embeddings
questions = [faq["question"] for faq in hr_faqs["faqs"]]
question_embeddings = sbert_model.encode(questions, convert_to_tensor=True)

# Define HR categories with updated topics
hr_categories = {
    "leave": ["What is the leave policy?", "How many leaves do I have left?", "Can I carry forward my unused leaves?"],
    "payroll": ["When will I get my salary?", "How do I check my salary slip?", "How do I update my bank account details for salary credit?"],
    "remote work": ["What is the work-from-home policy?", "Can I work remotely permanently?"],
    "performance": ["What are the promotion criteria?", "How do I enroll in company-sponsored training programs?"],
    "policies": ["What is the dress code policy?", "How do I report workplace harassment?", "How do I resign from the company?"]
}

# Compute embeddings for category questions
category_embeddings = {cat: sbert_model.encode(questions, convert_to_tensor=True) for cat, questions in hr_categories.items()}

def categorize_question(question):
    """Categorizes a user question based on SBERT similarity to predefined categories."""
    question_embedding = sbert_model.encode(question, convert_to_tensor=True)
    best_category = None
    highest_score = 0
    
    for category, embeddings in category_embeddings.items():
        similarity_scores = util.pytorch_cos_sim(question_embedding, embeddings)[0]
        max_score = torch.max(similarity_scores).item()
        if max_score > highest_score:
            highest_score = max_score
            best_category = category
    
    return best_category if highest_score > 0.75 else None  # Ensure strong categorization

# Memory to track user interactions
chat_memory = []
last_suggestion = None

# Initialize OpenAI Client for Project-Based API Keys
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_gpt_response(user_query):
    """Generates a response using OpenAI's API with a timeout."""
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an HR assistant helping employees with HR-related queries."},
                {"role": "user", "content": user_query}
            ],
            temperature=0.7,
            max_tokens=250,
            timeout=60
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] OpenAI API call failed: {str(e)}"

def get_answer(user_query):
    """Finds the best answer using SBERT, fuzzy matching, and GPT fallback."""
    global last_suggestion
    chat_memory.append(user_query)
    
    # Step 1: If user confirms previous suggestion
    if user_query.lower() in ["yes", "yeah", "y"] and last_suggestion:
        return hr_faqs["faqs"][questions.index(last_suggestion)]["answer"]
    
    # Step 2: Use SBERT for Semantic Similarity
    user_embedding = sbert_model.encode(user_query, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    sorted_indices = torch.argsort(cosine_scores, descending=True)[:5]  # Get top 5 closest
    
    best_sbert_index = sorted_indices[0].item()
    best_sbert_score = cosine_scores[best_sbert_index].item()
    
    # Step 3: Use Improved Fuzzy Matching
    fuzzy_results = process.extract(user_query, questions, scorer=fuzz.token_set_ratio, limit=5)
    fuzzy_matches = [res[0] for res in fuzzy_results if res[1] >= 75]  # Adjusted threshold
    
    # Step 4: Return the best match if confident enough
    if fuzzy_matches:
        last_suggestion = fuzzy_matches[0]  # Save suggestion for confirmation
        confidence_score = process.extractOne(user_query, questions, scorer=fuzz.token_set_ratio)[1]  # Get best match score

        if last_suggestion.lower() == user_query.lower():  # Direct match
            return hr_faqs["faqs"][questions.index(last_suggestion)]["answer"]

        if confidence_score >= 85:  # Only suggest if confidence is very high
            return f"Did you mean: '{last_suggestion}'?"
    
    # Step 5: If no good match, fall back to OpenAI GPT
    return get_gpt_response(user_query)

if __name__ == "__main__":
    print("HR Chatbot is running! Type your question or 'exit' to quit.")
    while True:
        user_input = input("Ask a question (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = get_answer(user_input)
        print("HR Chatbot:", response)
