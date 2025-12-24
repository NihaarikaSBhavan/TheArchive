# RAG App - The Archive

## Original Problem Statement
Build a RAG (Retrieval-Augmented Generation) app with:
- Multiple document formats (PDF, TXT, DOCX)
- Gemini LLM integration
- In-memory vector storage
- Chat history with source citations
- Aesthetic UI

## Architecture

### Backend (FastAPI)
- **Document Processing**: PyPDF2, python-docx for text extraction
- **Embeddings**: Gemini text-embedding-004 model
- **Vector Store**: In-memory with sklearn cosine similarity
- **LLM**: Gemini 2.0 Flash for response generation
- **Database**: MongoDB for documents metadata, chat sessions, messages

### Frontend (React)
- **Design**: Swiss Brutalist (0px borders, hard shadows)
- **Fonts**: Playfair Display (headings), Inter (body), JetBrains Mono (code)
- **Components**: Shadcn UI with custom brutalist styling
- **State**: React hooks with axios for API calls

### API Endpoints
- POST /api/documents/upload - Upload and process documents
- GET /api/documents - List all documents
- DELETE /api/documents/{id} - Delete document
- POST /api/chat/sessions - Create chat session
- GET /api/chat/sessions - List sessions
- DELETE /api/chat/sessions/{id} - Delete session
- GET /api/chat/sessions/{id}/messages - Get session messages
- POST /api/chat - Send message and get AI response

## Completed Tasks
1. ✅ Multi-format document upload (PDF, DOCX, TXT)
2. ✅ Text chunking with overlap
3. ✅ Gemini embeddings for semantic search
4. ✅ In-memory vector store with cosine similarity
5. ✅ Chat interface with history
6. ✅ Source citations in responses
7. ✅ MongoDB persistence for sessions/messages
8. ✅ Swiss Brutalist UI design
9. ✅ Error handling for API quota limits

## Next Tasks
1. Add drag-and-drop document upload
2. Add document preview functionality
3. Add document search/filtering
4. Add export chat history feature
5. Consider persistent vector store (MongoDB Atlas Vector Search)
6. Add user authentication for multi-user support

## Known Issues
- Gemini API quota limits may affect functionality
- Free tier has limited requests per day
