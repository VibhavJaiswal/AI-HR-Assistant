# AI-HR-Assistant

# AI-Powered HR Chatbot 🚀

## Overview
An **AI-powered HR Chatbot** that assists employees by answering HR-related queries using **Natural Language Processing (NLP)** and **Machine Learning**. This chatbot uses **FastAPI**, **SBERT (Sentence-BERT)**, **OpenAI GPT**, **ChromaDB**, and **Redis** for efficient and intelligent responses.

## Features 🔥
✅ **Smart HR Query Handling:** Uses **semantic search (SBERT) + fuzzy matching** to fetch relevant answers.
✅ **AI-Powered Answers:** If no direct match is found, queries are processed via **OpenAI GPT**.
✅ **FastAPI Integration:** The chatbot is deployed as a **RESTful API** with endpoints for querying HR policies.
✅ **ChromaDB Storage:** Vector database for improved HR FAQ retrieval.
✅ **Redis Memory:** Stores user interactions for context-aware responses.
✅ **Secure API Access:** Uses API keys to ensure authorized access.
✅ **Scalable & Modular:** Ready for integration into HRMS systems.

---

## Tech Stack 🛠️
- **Python 3.9+**
- **FastAPI** (For API deployment)
- **Sentence-BERT (SBERT)** (For semantic search)
- **OpenAI GPT-3.5 Turbo** (For AI-generated responses)
- **ChromaDB** (For vector-based FAQ retrieval)
- **Redis** (For session memory management)
- **RapidFuzz** (For fuzzy string matching)
- **Pydantic** (For data validation)
- **CORS Middleware** (For cross-origin support)

---

## Setup & Installation 🛠️
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/AI-HR-Assistant.git
cd AI-HR-Assistant
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up Environment Variables
Create a `.env` file and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_openai_api_key
```

### 4️⃣ Run the API Server
```bash
uvicorn hr_chatbot_api:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Endpoints 📡
| Endpoint             | Method | Description |
|----------------------|--------|-------------|
| `/chat`             | POST   | Process a user query and return an HR response |
| `/leave-balance`    | GET    | Fetch mock leave balance for a user |
| `/openapi.json`     | GET    | OpenAPI schema for API documentation |

#### Example Request (Chat Endpoint):
```bash
curl -X POST "http://localhost:8000/chat" -H "X-API-Key: your_api_key" -H "Content-Type: application/json" -d '{"query": "What is the leave policy?", "session_id": "user123"}'
```
#### Example Response:
```json
{
  "response": "The company offers X days of annual leave, Y sick leaves, and Z casual leaves. You can check your balance in the HR portal."
}
```

---

## File Structure 📁
```
├── hr_chatbot.py          # Main chatbot logic using SBERT & GPT
├── hr_chatbot_api.py      # FastAPI-based API server
├── expanded_hr_faq.json   # FAQ dataset for HR-related queries
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
```

---

## Future Enhancements 🚀
🔹 **Integration with HRMS systems** (SAP, Workday, BambooHR)  
🔹 **Voice-based chatbot support**  
🔹 **More advanced GPT fine-tuning**  
🔹 **Enhanced user authentication & role-based responses**  



