'use client';

import { useState, useEffect } from 'react';

interface CommunityConvo {
  id: string;
  title: string;
  contextType: string;
  contextExact: string;
  userName: string;
  createdAt: string;
  firstExchange: { role: string; content: string }[];
}

export default function CommunityConversations({ slug }: { slug: string }) {
  const [convos, setConvos] = useState<CommunityConvo[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/api/conversations/community?slug=${encodeURIComponent(slug)}`)
      .then((r) => r.ok ? r.json() : [])
      .then(setConvos)
      .catch(() => {});
  }, [slug]);

  if (convos.length === 0) return null;

  const openInChat = (conv: CommunityConvo) => {
    window.dispatchEvent(
      new CustomEvent('open-chat', {
        detail: {
          slug,
          contextType: conv.contextType,
          contextExact: conv.contextExact,
          conversationId: conv.id,
        },
      })
    );
  };

  return (
    <div className="community-conversations">
      <h3 className="community-conversations-title">Community conversations</h3>
      <div className="community-conversations-list">
        {convos.map((conv) => (
          <div
            key={conv.id}
            className={`community-conv-card ${expanded === conv.id ? 'expanded' : ''}`}
          >
            <button
              className="community-conv-header"
              onClick={() => setExpanded(expanded === conv.id ? null : conv.id)}
            >
              <span className="community-conv-title">{conv.title}</span>
              <span className="community-conv-meta">
                {conv.userName} · {conv.contextType}
              </span>
            </button>
            {expanded === conv.id && (
              <div className="community-conv-preview">
                {conv.firstExchange.map((msg, i) => (
                  <div key={i} className={`community-conv-msg community-conv-msg-${msg.role}`}>
                    <span className="community-conv-role">{msg.role === 'user' ? 'Q' : 'A'}:</span>
                    {' '}{msg.content.slice(0, 300)}{msg.content.length > 300 ? '...' : ''}
                  </div>
                ))}
                <button
                  className="community-conv-open"
                  onClick={() => openInChat(conv)}
                >
                  View full conversation
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
