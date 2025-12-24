from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Form
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
from emergentintegrations.llm.chat import LlmChat, UserMessage
from PyPDF2 import PdfReader
from docx import Document
import io
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import hashlib

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Emergent LLM Key
EMERGENT_KEY = os.environ['EMERGENT_LLM_KEY']

app = FastAPI()
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple text-based similarity (TF-IDF style) for embeddings without external API
def simple_embedding(text: str, dim: int = 256) -> List[float]:
    """Generate a simple hash-based embedding for text similarity."""
    # Normalize text
    text = text.lower().strip()
    words = re.findall(r'\w+', text)
    
    # Create embedding using word hashes
    embedding = [0.0] * dim
    for i, word in enumerate(words):
        # Hash word to get position and value
        hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
        pos = hash_val % dim
        value = ((hash_val >> 8) % 1000) / 1000.0
        embedding[pos] += value * (1.0 / (1 + i * 0.1))  # Position weighting
    
    # Normalize
    norm = sum(x*x for x in embedding) ** 0.5
    if norm > 0:
        embedding = [x / norm for x in embedding]
    
    return embedding

# In-memory vector store
class VectorStore:
    def __init__(self):
        self.documents = {}  # doc_id -> {chunks: [], embeddings: [], metadata: {}}
    
    def add_document(self, doc_id: str, chunks: List[str], embeddings: List[List[float]], metadata: dict):
        self.documents[doc_id] = {
            'chunks': chunks,
            'embeddings': embeddings,
            'metadata': metadata
        }
    
    def remove_document(self, doc_id: str):
        if doc_id in self.documents:
            del self.documents[doc_id]
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[dict]:
        results = []
        query_vec = np.array(query_embedding).reshape(1, -1)
        
        for doc_id, doc_data in self.documents.items():
            for i, emb in enumerate(doc_data['embeddings']):
                emb_vec = np.array(emb).reshape(1, -1)
                similarity = cosine_similarity(query_vec, emb_vec)[0][0]
                results.append({
                    'doc_id': doc_id,
                    'chunk': doc_data['chunks'][i],
                    'similarity': float(similarity),
                    'metadata': doc_data['metadata']
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

vector_store = VectorStore()

# Models
class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_type: str
    chunk_count: int
    uploaded_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    role: str  # 'user' or 'assistant'
    content: str
    citations: Optional[List[dict]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ChatSession(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    citations: List[dict]
    session_id: str

# Helper functions
def extract_text_from_pdf(file_bytes: bytes) -> str:
    pdf_reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks

# Routes
@api_router.get("/")
async def root():
    return {"message": "RAG API Ready"}

@api_router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename or "unknown"
        
        # Determine file type and extract text
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(content)
            file_type = 'pdf'
        elif filename.endswith('.docx'):
            text = extract_text_from_docx(content)
            file_type = 'docx'
        elif filename.endswith('.txt') or filename.endswith('.md'):
            text = content.decode('utf-8')
            file_type = 'txt'
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, DOCX, or TXT.")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the document.")
        
        # Chunk and embed
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid chunks could be created from the document.")
        
        # Use simple embedding (no external API needed)
        embeddings = [simple_embedding(chunk) for chunk in chunks]
        
        # Create document metadata
        doc_id = str(uuid.uuid4())
        metadata = DocumentMetadata(
            id=doc_id,
            filename=filename,
            file_type=file_type,
            chunk_count=len(chunks)
        )
        
        # Store in vector store
        vector_store.add_document(doc_id, chunks, embeddings, metadata.model_dump())
        
        # Store metadata in MongoDB
        await db.documents.insert_one(metadata.model_dump())
        
        return metadata.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/documents", response_model=List[DocumentMetadata])
async def list_documents():
    documents = await db.documents.find({}, {"_id": 0}).to_list(100)
    return documents

@api_router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    vector_store.remove_document(doc_id)
    result = await db.documents.delete_one({"id": doc_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted"}

@api_router.post("/chat/sessions")
async def create_chat_session(title: str = "New Chat"):
    session = ChatSession(title=title)
    await db.chat_sessions.insert_one(session.model_dump())
    return session.model_dump()

@api_router.get("/chat/sessions")
async def list_chat_sessions():
    sessions = await db.chat_sessions.find({}, {"_id": 0}).sort("updated_at", -1).to_list(50)
    return sessions

@api_router.get("/chat/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    messages = await db.chat_messages.find({"session_id": session_id}, {"_id": 0}).sort("timestamp", 1).to_list(100)
    return messages

@api_router.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    await db.chat_sessions.delete_one({"id": session_id})
    await db.chat_messages.delete_many({"session_id": session_id})
    return {"message": "Session deleted"}

@api_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Get query embedding and search for relevant chunks
        query_embedding = simple_embedding(request.message)
        relevant_chunks = vector_store.search(query_embedding, top_k=5)
        
        # Get chat history
        history = await db.chat_messages.find(
            {"session_id": request.session_id},
            {"_id": 0}
        ).sort("timestamp", 1).to_list(10)
        
        # Build context from relevant chunks
        context = ""
        citations = []
        seen_docs = set()
        
        for chunk in relevant_chunks:
            if chunk['similarity'] > 0.1:  # Lower threshold for simple embeddings
                context += f"\n---\nSource: {chunk['metadata']['filename']}\n{chunk['chunk']}\n"
                if chunk['doc_id'] not in seen_docs:
                    citations.append({
                        'doc_id': chunk['doc_id'],
                        'filename': chunk['metadata']['filename'],
                        'excerpt': chunk['chunk'][:200] + '...' if len(chunk['chunk']) > 200 else chunk['chunk'],
                        'similarity': round(chunk['similarity'], 3)
                    })
                    seen_docs.add(chunk['doc_id'])
        
        # Build chat history context
        history_context = ""
        for msg in history[-6:]:  # Last 6 messages
            role = "User" if msg['role'] == 'user' else "Assistant"
            history_context += f"{role}: {msg['content']}\n"
        
        # Build prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided documents.
If the context doesn't contain relevant information, say so clearly.
Always cite your sources when using information from the documents.
Be concise but thorough in your responses."""
        
        if context:
            user_prompt = f"""Previous conversation:
{history_context}

Relevant document excerpts:
{context}

User question: {request.message}

Provide a helpful answer based on the documents above. Reference which documents you're using."""
        else:
            user_prompt = f"""Previous conversation:
{history_context}

Note: No relevant documents found for this query. Please upload documents first or ask about uploaded content.

User question: {request.message}"""
        
        # Generate response using Emergent LLM
        llm_chat = LlmChat(
            api_key=EMERGENT_KEY,
            session_id=f"rag-{request.session_id}-{uuid.uuid4().hex[:8]}",
            system_message=system_prompt
        ).with_model("gemini", "gemini-2.5-flash")
        
        user_message = UserMessage(text=user_prompt)
        assistant_response = await llm_chat.send_message(user_message)
        
        # Save messages
        user_message_doc = ChatMessage(
            session_id=request.session_id,
            role="user",
            content=request.message
        )
        await db.chat_messages.insert_one(user_message_doc.model_dump())
        
        assistant_message = ChatMessage(
            session_id=request.session_id,
            role="assistant",
            content=assistant_response,
            citations=citations
        )
        await db.chat_messages.insert_one(assistant_message.model_dump())
        
        # Update session title if first message
        if len(history) == 0:
            new_title = request.message[:50] + "..." if len(request.message) > 50 else request.message
            await db.chat_sessions.update_one(
                {"id": request.session_id},
                {"$set": {"title": new_title, "updated_at": datetime.now(timezone.utc).isoformat()}}
            )
        else:
            await db.chat_sessions.update_one(
                {"id": request.session_id},
                {"$set": {"updated_at": datetime.now(timezone.utc).isoformat()}}
            )
        
        return ChatResponse(
            response=assistant_response,
            citations=citations,
            session_id=request.session_id
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
