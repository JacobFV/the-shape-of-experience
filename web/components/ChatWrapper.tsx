'use client';

import { useState, useEffect, useCallback } from 'react';
import { usePathname } from 'next/navigation';
import ChatPanel, { type ChatContext } from './ChatPanel';

interface OpenChatEvent extends CustomEvent {
  detail: ChatContext;
}

export default function ChatWrapper() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [context, setContext] = useState<ChatContext>({
    contextType: 'book',
  });

  const slug = pathname.replace(/^\//, '') || undefined;

  const handleOpenChat = useCallback((e: Event) => {
    const detail = (e as OpenChatEvent).detail;
    setContext({
      slug: detail.slug || slug,
      contextType: detail.contextType || 'page',
      contextExact: detail.contextExact,
      contextHeadingId: detail.contextHeadingId,
    });
    setOpen(true);
  }, [slug]);

  useEffect(() => {
    window.addEventListener('open-chat', handleOpenChat);
    return () => window.removeEventListener('open-chat', handleOpenChat);
  }, [handleOpenChat]);

  return (
    <ChatPanel
      open={open}
      onClose={() => setOpen(false)}
      context={context}
    />
  );
}
