import { useState, useEffect, useRef } from 'react';
import '@/App.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import axios from 'axios';
import { Toaster, toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  Upload,
  Send,
  FileText,
  MessageSquare,
  Plus,
  Trash2,
  File,
  Loader2,
  BookOpen,
  X,
} from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Document Sidebar Component
const DocumentSidebar = ({ documents, onUpload, onDelete, isUploading }) => {
  const fileInputRef = useRef(null);

  const handleFileSelect = async (e) => {
    const file = e.target.files?.[0];
    if (file) {
      await onUpload(file);
      e.target.value = '';
    }
  };

  return (
    <div className="w-72 border-r border-border bg-sidebar h-full flex flex-col" data-testid="document-sidebar">
      <div className="p-6 border-b border-border">
        <h2 className="font-heading text-lg font-bold tracking-tight mb-4">DOCUMENTS</h2>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileSelect}
          accept=".pdf,.docx,.txt,.md"
          className="hidden"
          data-testid="file-input"
        />
        <Button
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
          className="w-full bg-primary text-primary-foreground btn-brutalist border border-foreground"
          data-testid="upload-btn"
        >
          {isUploading ? (
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
          ) : (
            <Upload className="h-4 w-4 mr-2" />
          )}
          {isUploading ? 'Processing...' : 'Upload Document'}
        </Button>
        <p className="text-xs text-muted-foreground mt-2 font-mono tracking-wide">
          PDF, DOCX, TXT supported
        </p>
      </div>

      <ScrollArea className="flex-1 p-4">
        {documents.length === 0 ? (
          <div className="text-center py-12">
            <div className="w-16 h-16 mx-auto mb-4 border border-border flex items-center justify-center">
              <FileText className="h-8 w-8 text-muted-foreground" strokeWidth={1.5} />
            </div>
            <p className="text-sm text-muted-foreground">No documents yet</p>
            <p className="text-xs text-muted-foreground mt-1">Upload to start</p>
          </div>
        ) : (
          <div className="space-y-2">
            {documents.map((doc, idx) => (
              <div
                key={doc.id}
                className={`p-4 border border-border bg-card card-brutalist group animate-fadeInUp stagger-${Math.min(idx + 1, 5)}`}
                style={{ opacity: 0 }}
                data-testid={`document-${doc.id}`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <File className="h-4 w-4 text-primary flex-shrink-0" strokeWidth={1.5} />
                      <p className="text-sm font-medium truncate">{doc.filename}</p>
                    </div>
                    <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground font-mono">
                      <span className="uppercase">{doc.file_type}</span>
                      <span>•</span>
                      <span>{doc.chunk_count} chunks</span>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => onDelete(doc.id)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity h-8 w-8"
                    data-testid={`delete-doc-${doc.id}`}
                  >
                    <Trash2 className="h-4 w-4 text-destructive" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  );
};

// Chat Sessions Sidebar Component
const ChatSidebar = ({ sessions, currentSession, onSelect, onCreate, onDelete }) => (
  <div className="w-64 border-r border-border h-full flex flex-col" data-testid="chat-sidebar">
    <div className="p-6 border-b border-border">
      <h2 className="font-heading text-lg font-bold tracking-tight mb-4">CHAT HISTORY</h2>
      <Button
        onClick={onCreate}
        variant="outline"
        className="w-full btn-brutalist border-foreground"
        data-testid="new-chat-btn"
      >
        <Plus className="h-4 w-4 mr-2" />
        New Chat
      </Button>
    </div>

    <ScrollArea className="flex-1 p-4">
      {sessions.length === 0 ? (
        <div className="text-center py-12">
          <MessageSquare className="h-8 w-8 mx-auto text-muted-foreground mb-2" strokeWidth={1.5} />
          <p className="text-sm text-muted-foreground">No chats yet</p>
        </div>
      ) : (
        <div className="space-y-2">
          {sessions.map((session) => (
            <div
              key={session.id}
              onClick={() => onSelect(session.id)}
              className={`p-3 border cursor-pointer transition-colors group ${
                currentSession === session.id
                  ? 'border-primary bg-muted'
                  : 'border-border hover:border-primary'
              }`}
              data-testid={`session-${session.id}`}
            >
              <div className="flex items-center justify-between">
                <p className="text-sm truncate flex-1">{session.title}</p>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(session.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 h-6 w-6"
                  data-testid={`delete-session-${session.id}`}
                >
                  <X className="h-3 w-3" />
                </Button>
              </div>
              <p className="text-xs text-muted-foreground font-mono mt-1">
                {new Date(session.updated_at).toLocaleDateString()}
              </p>
            </div>
          ))}
        </div>
      )}
    </ScrollArea>
  </div>
);

// Citation Card Component
const CitationCard = ({ citation, index }) => (
  <div
    className="p-4 border border-border bg-card mt-2 animate-fadeInUp"
    style={{ animationDelay: `${index * 0.05}s`, opacity: 0 }}
    data-testid={`citation-${citation.doc_id}`}
  >
    <div className="flex items-center gap-2 mb-2">
      <BookOpen className="h-4 w-4 text-primary" strokeWidth={1.5} />
      <span className="text-xs font-mono text-muted-foreground uppercase tracking-widest">
        Source [{index + 1}]
      </span>
    </div>
    <p className="text-sm font-medium mb-1">{citation.filename}</p>
    <p className="text-xs text-muted-foreground font-mono leading-relaxed">
      "{citation.excerpt}"
    </p>
    <p className="text-xs text-primary font-mono mt-2">
      Relevance: {Math.round(citation.similarity * 100)}%
    </p>
  </div>
);

// Chat Message Component
const ChatMessage = ({ message }) => (
  <div
    className={`mb-6 animate-fadeInUp ${
      message.role === 'user' ? 'chat-bubble-user' : 'chat-bubble-assistant'
    }`}
    style={{ opacity: 0 }}
    data-testid={`message-${message.id}`}
  >
    <div className="flex items-center gap-2 mb-2">
      <span className="text-xs font-mono uppercase tracking-widest text-muted-foreground">
        {message.role === 'user' ? 'You' : 'Archive'}
      </span>
      <span className="text-xs font-mono text-muted-foreground">
        {new Date(message.timestamp).toLocaleTimeString()}
      </span>
    </div>
    <div className="prose prose-sm max-w-none">
      <p className="text-foreground leading-relaxed whitespace-pre-wrap">{message.content}</p>
    </div>
    {message.citations && message.citations.length > 0 && (
      <div className="mt-4 pt-4 border-t border-border">
        <p className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-2">
          Citations
        </p>
        {message.citations.map((citation, idx) => (
          <CitationCard key={citation.doc_id} citation={citation} index={idx} />
        ))}
      </div>
    )}
  </div>
);

// Main Chat Interface
const ChatInterface = ({ sessionId, messages, onSend, isLoading }) => {
  const [input, setInput] = useState('');
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSend(input.trim());
      setInput('');
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-background bg-noise" data-testid="chat-interface">
      {!sessionId ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center p-12 max-w-md">
            <h1 className="font-heading text-4xl font-bold tracking-tight mb-4">The Archive</h1>
            <p className="text-muted-foreground mb-8">
              Upload documents and ask questions. Your AI research assistant with source citations.
            </p>
            <div className="border border-border p-6 bg-card">
              <p className="text-sm text-muted-foreground font-mono">
                1. Upload documents (PDF, DOCX, TXT)<br />
                2. Start a new chat<br />
                3. Ask questions about your documents
              </p>
            </div>
          </div>
        </div>
      ) : (
        <>
          <ScrollArea className="flex-1 p-8" ref={scrollRef}>
            <div className="max-w-3xl mx-auto relative z-10">
              {messages.length === 0 ? (
                <div className="text-center py-20">
                  <MessageSquare className="h-12 w-12 mx-auto text-muted-foreground mb-4" strokeWidth={1} />
                  <p className="text-muted-foreground">Start asking questions about your documents</p>
                </div>
              ) : (
                messages.map((msg) => <ChatMessage key={msg.id} message={msg} />)
              )}
              {isLoading && (
                <div className="chat-bubble-assistant mb-6 animate-pulse">
                  <div className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin text-primary" />
                    <span className="text-sm text-muted-foreground">Searching documents...</span>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>

          <div className="border-t border-border p-6 bg-card relative z-10">
            <form onSubmit={handleSubmit} className="max-w-3xl mx-auto flex gap-4">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about your documents..."
                disabled={isLoading}
                className="flex-1 input-brutalist text-base py-6"
                data-testid="chat-input"
              />
              <Button
                type="submit"
                disabled={!input.trim() || isLoading}
                className="bg-primary text-primary-foreground btn-brutalist border border-foreground px-6"
                data-testid="send-btn"
              >
                {isLoading ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <Send className="h-5 w-5" />
                )}
              </Button>
            </form>
          </div>
        </>
      )}
    </div>
  );
};

// Main App Component
const RAGApp = () => {
  const [documents, setDocuments] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [currentSession, setCurrentSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Fetch documents and sessions on mount
  useEffect(() => {
    fetchDocuments();
    fetchSessions();
  }, []);

  // Fetch messages when session changes
  useEffect(() => {
    if (currentSession) {
      fetchMessages(currentSession);
    } else {
      setMessages([]);
    }
  }, [currentSession]);

  const fetchDocuments = async () => {
    try {
      const response = await axios.get(`${API}/documents`);
      setDocuments(response.data);
    } catch (error) {
      console.error('Error fetching documents:', error);
    }
  };

  const fetchSessions = async () => {
    try {
      const response = await axios.get(`${API}/chat/sessions`);
      setSessions(response.data);
    } catch (error) {
      console.error('Error fetching sessions:', error);
    }
  };

  const fetchMessages = async (sessionId) => {
    try {
      const response = await axios.get(`${API}/chat/sessions/${sessionId}/messages`);
      setMessages(response.data);
    } catch (error) {
      console.error('Error fetching messages:', error);
    }
  };

  const handleUpload = async (file) => {
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      await axios.post(`${API}/documents/upload`, formData);
      toast.success('Document uploaded and processed!');
      fetchDocuments();
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDeleteDocument = async (docId) => {
    try {
      await axios.delete(`${API}/documents/${docId}`);
      toast.success('Document deleted');
      fetchDocuments();
    } catch (error) {
      toast.error('Failed to delete document');
    }
  };

  const handleCreateSession = async () => {
    try {
      const response = await axios.post(`${API}/chat/sessions`);
      setSessions([response.data, ...sessions]);
      setCurrentSession(response.data.id);
    } catch (error) {
      toast.error('Failed to create chat');
    }
  };

  const handleDeleteSession = async (sessionId) => {
    try {
      await axios.delete(`${API}/chat/sessions/${sessionId}`);
      setSessions(sessions.filter((s) => s.id !== sessionId));
      if (currentSession === sessionId) {
        setCurrentSession(null);
        setMessages([]);
      }
    } catch (error) {
      toast.error('Failed to delete chat');
    }
  };

  const handleSendMessage = async (content) => {
    if (!currentSession) return;

    setIsLoading(true);
    // Optimistically add user message
    const tempUserMsg = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, tempUserMsg]);

    try {
      const response = await axios.post(`${API}/chat`, {
        session_id: currentSession,
        message: content,
      });

      // Replace temp message and add assistant response
      fetchMessages(currentSession);
      fetchSessions(); // Update session title
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Failed to send message');
      // Remove optimistic message on error
      setMessages((prev) => prev.filter((m) => m.id !== tempUserMsg.id));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-screen flex" data-testid="rag-app">
      <DocumentSidebar
        documents={documents}
        onUpload={handleUpload}
        onDelete={handleDeleteDocument}
        isUploading={isUploading}
      />
      <ChatSidebar
        sessions={sessions}
        currentSession={currentSession}
        onSelect={setCurrentSession}
        onCreate={handleCreateSession}
        onDelete={handleDeleteSession}
      />
      <ChatInterface
        sessionId={currentSession}
        messages={messages}
        onSend={handleSendMessage}
        isLoading={isLoading}
      />
      <Toaster position="bottom-right" />
    </div>
  );
};

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<RAGApp />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
