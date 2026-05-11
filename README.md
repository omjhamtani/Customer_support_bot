# Goodluck Cafe Customer Support Bot

An AI-powered backend for a customer support chatbot, specifically tailored for "Goodluck Cafe in Pune". This project uses a **Retrieval-Augmented Generation (RAG)** approach to provide accurate, context-aware answers to user queries based on a local knowledge base.

## 🚀 Features

- **FastAPI Backend:** A fast, modern web framework for building the API endpoints.
- **RAG Architecture:** Leverages LangChain to retrieve relevant information before generating a response.
- **Google Gemini Integration:** Uses `gemini-pro` for text generation and `models/embedding-001` for vector embeddings.
- **FAISS Vector Store:** An in-memory vector database used to quickly search through the cafe's knowledge base.
- **Hallucination Prevention:** Strict prompting ensures the bot only answers using the provided context and gracefully declines unknown queries.
- **CORS Enabled:** Ready to be consumed by any frontend application.

## 🛠️ Tech Stack

- **Python 3.x**
- **FastAPI**
- **LangChain** & **LangChain Google GenAI**
- **FAISS** (Facebook AI Similarity Search)
- **Uvicorn** (ASGI server)

## 📦 Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/omjhamtani/Customer_support_bot.git
   cd Customer_support_bot
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   Make sure you have the required packages installed:
   ```bash
   pip install fastapi uvicorn langchain langchain-google-genai faiss-cpu python-dotenv pydantic
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your Google API Key:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. **Knowledge Base:**
   Ensure you have a `knowledge_base.md` file in the root directory containing the cafe's menus, rules, FAQs, and general information. This file is parsed on application startup to build the vector store.

## 🚦 Running the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`. You can explore the interactive API documentation (Swagger UI) at `http://localhost:8000/docs`.

## 🔌 API Endpoints

### `POST /process-message`

Processes a user's question and returns an AI-generated response based on the knowledge base.

**Request Body:**
```json
{
  "query": "What are your opening hours?"
}
```

**Response:**
```json
{
  "reply": "We are open from 8 AM to 10 PM every day."
}
```