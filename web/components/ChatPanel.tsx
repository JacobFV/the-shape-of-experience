'use client';

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import { useSession } from 'next-auth/react';
import { useConversations, type Conversation } from '@/lib/hooks/useConversations';

export interface ChatContext {
  slug?: string;
  contextType: 'highlight' | 'page' | 'book';
  contextExact?: string;
  contextHeadingId?: string;
}

interface ChatPanelProps {
  open: boolean;
  onClose: () => void;
  context: ChatContext;
}

function getMessageText(msg: { parts?: Array<{ type: string; text?: string }>; content?: string }): string {
  if (msg.parts) {
    return msg.parts
      .filter((p) => p.type === 'text' && p.text)
      .map((p) => p.text!)
      .join('');
  }
  return msg.content || '';
}

export default function ChatPanel({ open, onClose, context }: ChatPanelProps) {
  const { data: session, status: authStatus } = useSession();
  const isAuth = authStatus === 'authenticated' && !!session?.user;
  const { items: conversations, refresh, updateConversation, removeConversation } = useConversations(context.slug);
  const [activeConvId, setActiveConvId] = useState<string | null>(null);
  const [showList, setShowList] = useState(false);
  const [loadingConv, setLoadingConv] = useState(false);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const transport = useMemo(() => new DefaultChatTransport({
    api: '/api/chat',
    body: {
      conversationId: activeConvId,
      contextType: context.contextType,
      slug: context.slug,
      contextExact: context.contextExact,
      contextHeadingId: context.contextHeadingId,
    },
  }), [activeConvId, context.contextType, context.slug, context.contextExact, context.contextHeadingId]);

  const { messages, sendMessage, setMessages, status } = useChat({
    transport,
    onFinish: ({ message }) => {
      // After first response, refresh conversation list to pick up new conversation
      if (!activeConvId) {
        refresh();
      }
    },
  });

  const isLoading = status === 'submitted' || status === 'streaming';

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input when panel opens
  useEffect(() => {
    if (open) setTimeout(() => inputRef.current?.focus(), 300);
  }, [open]);

  // Reset when context changes
  useEffect(() => {
    setActiveConvId(null);
    setMessages([]);
    setShowList(false);
    setInput('');
  }, [context.contextType, context.contextExact, context.slug, setMessages]);

  const loadConversation = useCallback(async (conv: Conversation) => {
    setLoadingConv(true);
    setShowList(false);
    try {
      const res = await fetch(`/api/conversations/${conv.id}`);
      if (res.ok) {
        const data = await res.json();
        setActiveConvId(conv.id);
        setMessages(
          data.messages.map((m: { id: string; role: string; content: string }) => ({
            id: m.id,
            role: m.role as 'user' | 'assistant',
            parts: [{ type: 'text' as const, text: m.content }],
          }))
        );
      }
    } catch { /* ignore */ }
    setLoadingConv(false);
  }, [setMessages]);

  const startNew = useCallback(() => {
    setActiveConvId(null);
    setMessages([]);
    setShowList(false);
    setInput('');
  }, [setMessages]);

  const togglePublish = useCallback(async () => {
    if (!activeConvId) return;
    const conv = conversations.find((c) => c.id === activeConvId);
    if (!conv) return;
    await updateConversation(activeConvId, { isPublished: !conv.isPublished });
  }, [activeConvId, conversations, updateConversation]);

  const deleteConversation = useCallback(async () => {
    if (!activeConvId) return;
    await removeConversation(activeConvId);
    startNew();
    refresh();
  }, [activeConvId, removeConversation, startNew, refresh]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || isLoading) return;
    setInput('');
    await sendMessage({ text });
  }, [input, isLoading, sendMessage]);

  const activeConv = conversations.find((c) => c.id === activeConvId);

  const contextLabel =
    context.contextType === 'highlight'
      ? `"${(context.contextExact || '').slice(0, 40)}..."`
      : context.contextType === 'page'
        ? context.slug?.replace(/-/g, ' ') || 'this page'
        : 'the full book';

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (!open) return null;

  return (
    <div className="chat-panel-overlay" onClick={onClose}>
      <div className="chat-panel" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="chat-panel-header">
          <div className="chat-panel-header-left">
            <button
              className="chat-panel-btn"
              onClick={() => setShowList(!showList)}
              title="Conversations"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M3 12h18M3 6h18M3 18h18" />
              </svg>
            </button>
            <span className="chat-panel-title">
              {activeConv?.title || 'New chat'}
            </span>
          </div>
          <div className="chat-panel-header-right">
            {activeConvId && isAuth && (
              <>
                <button
                  className={`chat-panel-btn chat-publish-btn ${activeConv?.isPublished ? 'published' : ''}`}
                  onClick={togglePublish}
                  title={activeConv?.isPublished ? 'Unpublish' : 'Publish for community'}
                >
                  {activeConv?.isPublished ? 'Public' : 'Publish'}
                </button>
                <button
                  className="chat-panel-btn"
                  onClick={deleteConversation}
                  title="Delete conversation"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2" />
                  </svg>
                </button>
              </>
            )}
            <button className="chat-panel-btn" onClick={onClose} title="Close">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M18 6L6 18M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Conversation list */}
        {showList && (
          <div className="chat-conv-list">
            <button className="chat-conv-item chat-conv-new" onClick={startNew}>
              + New conversation
            </button>
            {conversations.map((conv) => (
              <button
                key={conv.id}
                className={`chat-conv-item ${conv.id === activeConvId ? 'active' : ''}`}
                onClick={() => loadConversation(conv)}
              >
                <span className="chat-conv-title">{conv.title}</span>
                <span className="chat-conv-meta">
                  {conv.contextType}
                  {conv.isPublished && ' · public'}
                </span>
              </button>
            ))}
            {conversations.length === 0 && (
              <div className="chat-conv-empty">No conversations yet</div>
            )}
          </div>
        )}

        {/* Auth gate */}
        {!isAuth ? (
          <div className="chat-auth-gate">
            <p>Sign in to chat about the book.</p>
            <p className="chat-auth-hint">Use the account button in the toolbar above.</p>
          </div>
        ) : loadingConv ? (
          <div className="chat-loading">Loading...</div>
        ) : (
          <>
            {/* Messages */}
            <div className="chat-messages">
              {messages.length === 0 && (
                <div className="chat-empty">
                  <p className="chat-empty-title">Ask about {contextLabel}</p>
                  <p className="chat-empty-hint">
                    Ask questions, explore ideas, or discuss concepts from the book.
                  </p>
                </div>
              )}
              {messages.map((msg) => (
                <div key={msg.id} className={`chat-message chat-message-${msg.role}`}>
                  <div className="chat-message-content">{getMessageText(msg)}</div>
                </div>
              ))}
              {isLoading && messages.length > 0 && messages[messages.length - 1]?.role === 'user' && (
                <div className="chat-message chat-message-assistant">
                  <div className="chat-message-content chat-typing">Thinking...</div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="chat-input-area">
              <div className="chat-context-badge">
                {context.contextType === 'highlight' && 'Highlight'}
                {context.contextType === 'page' && 'Page'}
                {context.contextType === 'book' && 'Book'}
              </div>
              <div className="chat-input-form">
                <textarea
                  ref={inputRef}
                  className="chat-input"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={onKeyDown}
                  placeholder="Ask a question..."
                  rows={1}
                  disabled={isLoading}
                />
                <button
                  type="button"
                  className="chat-send-btn"
                  disabled={isLoading || !input.trim()}
                  onClick={handleSend}
                  title="Send"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                  </svg>
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
