# main.py

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware # <--- ADD THIS LINE
from dotenv import load_dotenv

# LangChain components
# ... (rest of the imports)
# LangChain components
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- 1. INITIAL SETUP ---

# Load environment variables from a .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check for API key
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

# Initialize the FastAPI app
# main.py

# ... (after imports)

# Initialize the FastAPI app
app = FastAPI(
    title="Goodluck Cafe Support Bot",
    description="An AI assistant for Goodluck Cafe based on a knowledge base.",
    version="1.0.0"
)

# --- ADD THIS ENTIRE BLOCK ---
# This allows your frontend to communicate with your backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# --- END OF BLOCK ---

# --- 2. STARTUP EVENT: LOAD AND PROCESS KNOWLEDGE BASE ---
# ... (rest of your file)

# In-memory storage for our vector store
# This will be populated at startup
vector_store = None

# --- 2. STARTUP EVENT: LOAD AND PROCESS KNOWLEDGE BASE ---
# This code runs only once when the application starts.

@app.on_event("startup")
def load_knowledge_base():
    global vector_store

    loader = None
    try:
        with open("knowledge_base.md", "r", encoding="utf-8") as f:
            knowledge_base_text = f.read()
    except FileNotFoundError:
        raise RuntimeError("knowledge_base.md not found. Please ensure the file exists.")

    # Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_text(knowledge_base_text)

    # Initialize Google embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    # Create the FAISS vector store from the document chunks
    # This process creates numerical representations (embeddings) of our text
    # and stores them in an efficient index for searching.
    print("Creating vector store... this may take a moment.")
    vector_store = FAISS.from_texts(docs, embeddings)
    print("Vector store created successfully.")


# --- 3. API DATA MODELS ---
# Pydantic models for request and response validation

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    reply: str

# --- 4. API ENDPOINT TO PROCESS QUERIES ---

@app.post("/process-message", response_model=QueryResponse)
async def process_message(request: QueryRequest):
    global vector_store

    if not vector_store:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded yet. Please try again in a moment.")

    # Initialize the LLM for chat responses
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.3)

    # Create the prompt template
    prompt_template = """
    You are a friendly and helpful customer support assistant for Goodluck Cafe in Pune.
    Answer the user's question based ONLY on the following context.
    If the information is not in the context, you MUST politely say 'I'm sorry, I don't have information on that.'.
    Do not make up any information. Keep your answers concise and to the point.

    CONTEXT:
    {context}

    USER'S QUESTION:
    {question}

    YOUR ANSWER:
    """

    # Create the RetrievalQA chain
    # This chain does the following:
    # 1. Takes the user's query.
    # 2. Searches the vector_store for relevant context (the "Retrieval" part).
    # 3. Puts the context and query into the prompt.
    # 4. Sends it to the LLM to generate an answer (the "QA" part).
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": {
            "template": prompt_template,
            "input_variables": ["context", "question"]
        }}
    )

    try:
        # Run the chain with the user's query
        result = qa_chain.invoke({"query": request.query})
        return QueryResponse(reply=result["result"])
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")
