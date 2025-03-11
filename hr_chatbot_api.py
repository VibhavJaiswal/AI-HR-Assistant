from fastapi import FastAPI, HTTPException, Depends, Query, Security
from pydantic import BaseModel
import json
import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import os
from rapidfuzz import process, fuzz
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
import redis
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.openapi.utils import get_openapi
import chromadb

# Initialize Redis for session memory
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Initialize ChromaDB for HR FAQ vector storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")
hr_faq_collection = chroma_client.get_or_create_collection(name="hr_faqs")

# Load HR FAQs and store them in ChromaDB
FAQ_FILE = "expanded_hr_faq.json"
try:
    with open(FAQ_FILE, "r", encoding="utf-8") as file:
        hr_faqs = json.load(file)
except FileNotFoundError:
    raise Exception(f"Error: '{FAQ_FILE}' file not found. Make sure it is in the same directory as this script.")

for i, faq in enumerate(hr_faqs["faqs"]):
    question = faq["question"]
    answer = faq["answer"]
    existing_docs = hr_faq_collection.get(where={"question": question})
    if not existing_docs.get("documents"):
        hr_faq_collection.add(
            ids=[str(i)],
            documents=[question],
            metadatas=[{"answer": answer}]
        )

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# API Key Security
API_KEY = "my_secret_api_key"  # Change this to a secure key
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized API key")
    return api_key

# FastAPI Initialization
app = FastAPI(title="HR Chatbot API", version="1.0", description="An AI-powered HR chatbot API with session memory.")

# Enable CORS (for frontend integration if needed in the future)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom OpenAPI function to add API key security

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="HR Chatbot API",
        version="1.0",
        description="An AI-powered HR chatbot API with session memory.",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        API_KEY_NAME: {
            "type": "apiKey",
            "in": "header",
            "name": API_KEY_NAME,
        }
    }
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            openapi_schema["paths"][path][method]["security"] = [{API_KEY_NAME: []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Request Model
class ChatRequest(BaseModel):
    query: str
    session_id: str  # Unique session identifier

# Simulated HRMS leave balance data
class LeaveBalanceResponse(BaseModel):
    session_id: str
    leave_balance: int

mock_employee_data = {
    "test123": {"leave_balance": 12},  # Example session_id -> leave balance
    "emp456": {"leave_balance": 5}
}
mock_employee_data["test123"] = {"leave_balance": 12}

@app.get("/leave-balance", tags=["HRMS"], response_model=LeaveBalanceResponse)
async def get_leave_balance(session_id: str = Query(..., description="Session ID of the user"), api_key: APIKey = Depends(verify_api_key)):
    """Mock API to fetch leave balance for a user."""
    print(f"[DEBUG] Received session_id: {session_id}")  # Debugging print statement
    if session_id in mock_employee_data:
        return LeaveBalanceResponse(session_id=session_id, leave_balance=mock_employee_data[session_id]["leave_balance"])
    else:
        print("[DEBUG] Employee not found in mock data")  # Debugging print
        raise HTTPException(status_code=404, detail="Employee not found")

@app.get("/openapi.json", include_in_schema=True)
def get_openapi_json():
    """Ensure OpenAPI schema is accessible."""
    return app.openapi()

# Chatbot Logic

def get_answer(user_query, session_id):
    """Finds the best HR response using ChromaDB for semantic search."""
    print("\n[DEBUG] Searching ChromaDB for:", user_query)
    
    results = hr_faq_collection.query(
        query_texts=[user_query],
        n_results=3  # Get top 3 matches
    )
    
    if "documents" in results and results["documents"] and len(results["documents"][0]) > 0:
        best_match = results["documents"][0][0]  # Extract first result correctly
        best_answer = results["metadatas"][0][0]["answer"]  # Extract answer properly
        print("[DEBUG] Best Match Found:", best_match)
        return best_answer

    print("[DEBUG] No confident match. Using GPT-3.5")
    return get_gpt_response(user_query)

# API Endpoints
@app.post("/chat")
def chat(request: ChatRequest, api_key: APIKey = Depends(verify_api_key)):
    try:
        response = get_answer(request.query, request.session_id)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Welcome to the HR Chatbot API! Use /chat endpoint to ask questions with session memory."}
