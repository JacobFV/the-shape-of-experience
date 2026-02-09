'use client';

import { useEffect, useRef, useCallback, useState } from 'react';

interface HighlightData {
  id: string;
  nearestHeadingId: string;
  prefix: string;
  exact: string;
  suffix: string;
  createdAt: number;
}

function getHighlights(slug: string): HighlightData[] {
  try {
    return JSON.parse(localStorage.getItem(`soe-highlights-${slug}`) || '[]');
  } catch { return []; }
}

function saveHighlights(slug: string, highlights: HighlightData[]) {
  localStorage.setItem(`soe-highlights-${slug}`, JSON.stringify(highlights));
}

function findNearestHeadingId(node: Node): string {
  let el: Element | null = node.nodeType === Node.TEXT_NODE ? node.parentElement : node as Element;
  while (el) {
    // Walk backward through siblings to find a heading
    let prev: Element | null = el.previousElementSibling;
    while (prev) {
      if (/^H[1-3]$/i.test(prev.tagName) && prev.id) return prev.id;
      prev = prev.previousElementSibling;
    }
    el = el.parentElement;
  }
  return '';
}

function getContext(range: Range): { prefix: string; exact: string; suffix: string } {
  const exact = range.toString();
  const container = range.startContainer;
  const text = container.textContent || '';
  const start = range.startOffset;
  const end = range.endOffset;
  const prefix = text.slice(Math.max(0, start - 30), start);
  const suffix = text.slice(end, end + 30);
  return { prefix, exact, suffix };
}

function findTextInContent(
  root: HTMLElement,
  highlight: HighlightData
): Range | null {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let node: Text | null;
  while ((node = walker.nextNode() as Text | null)) {
    const text = node.textContent || '';
    const idx = text.indexOf(highlight.exact);
    if (idx === -1) continue;

    // Verify context if available
    if (highlight.prefix) {
      const before = text.slice(Math.max(0, idx - 30), idx);
      if (!before.includes(highlight.prefix.slice(-10))) continue;
    }

    const range = document.createRange();
    range.setStart(node, idx);
    range.setEnd(node, idx + highlight.exact.length);
    return range;
  }
  return null;
}

export default function HighlightManager({ slug }: { slug: string }) {
  const [popover, setPopover] = useState<{ x: number; y: number; range: Range } | null>(null);
  const [removePopover, setRemovePopover] = useState<{ x: number; y: number; id: string } | null>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
  const restoredRef = useRef(false);

  // Restore highlights on mount
  useEffect(() => {
    if (restoredRef.current) return;
    restoredRef.current = true;

    // Small delay so DOM is ready
    const timer = setTimeout(() => {
      const content = document.querySelector('.chapter-content');
      if (!content) return;
      const highlights = getHighlights(slug);
      for (const h of highlights) {
        const range = findTextInContent(content as HTMLElement, h);
        if (range) {
          const mark = document.createElement('mark');
          mark.className = 'user-highlight';
          mark.dataset.highlightId = h.id;
          try { range.surroundContents(mark); } catch { /* partial overlap */ }
        }
      }
    }, 500);
    return () => clearTimeout(timer);
  }, [slug]);

  // Handle text selection
  const onMouseUp = useCallback((e: MouseEvent) => {
    // Ignore clicks on existing marks/popover
    const target = e.target as HTMLElement;
    if (target.closest('.highlight-popover') || target.closest('.reader-toolbar')) return;

    // Check for click on existing highlight
    if (target.closest('mark.user-highlight')) {
      const mark = target.closest('mark.user-highlight') as HTMLElement;
      const id = mark.dataset.highlightId;
      if (id) {
        setRemovePopover({ x: e.clientX, y: e.clientY - 40, id });
        setPopover(null);
        return;
      }
    }

    setRemovePopover(null);

    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.rangeCount) {
      setPopover(null);
      return;
    }

    const range = sel.getRangeAt(0);
    const text = range.toString().trim();
    if (text.length < 3) { setPopover(null); return; }

    // Only for chapter content
    const content = document.querySelector('.chapter-content');
    if (!content || !content.contains(range.commonAncestorContainer)) {
      setPopover(null);
      return;
    }

    const rect = range.getBoundingClientRect();
    setPopover({
      x: rect.left + rect.width / 2,
      y: rect.top - 44,
      range: range.cloneRange(),
    });
  }, []);

  useEffect(() => {
    document.addEventListener('mouseup', onMouseUp);
    return () => document.removeEventListener('mouseup', onMouseUp);
  }, [onMouseUp]);

  const doHighlight = useCallback(() => {
    if (!popover) return;
    const range = popover.range;
    const ctx = getContext(range);
    const headingId = findNearestHeadingId(range.startContainer);

    const h: HighlightData = {
      id: `hl-${Date.now()}`,
      nearestHeadingId: headingId,
      prefix: ctx.prefix,
      exact: ctx.exact,
      suffix: ctx.suffix,
      createdAt: Date.now(),
    };

    // Apply mark
    const mark = document.createElement('mark');
    mark.className = 'user-highlight';
    mark.dataset.highlightId = h.id;
    try { range.surroundContents(mark); } catch { /* partial overlap */ }

    // Save
    const highlights = [...getHighlights(slug), h];
    saveHighlights(slug, highlights);

    window.getSelection()?.removeAllRanges();
    setPopover(null);
  }, [popover, slug]);

  const doShare = useCallback(() => {
    if (!popover) return;
    const text = popover.range.toString().trim();
    // Text Fragments API
    const encoded = encodeURIComponent(text.slice(0, 100));
    const url = `${window.location.origin}/${slug}#:~:text=${encoded}`;
    navigator.clipboard.writeText(url).catch(() => {});
    window.getSelection()?.removeAllRanges();
    setPopover(null);
  }, [popover, slug]);

  const doRemoveHighlight = useCallback(() => {
    if (!removePopover) return;
    const { id } = removePopover;
    // Remove mark from DOM
    const mark = document.querySelector(`mark[data-highlight-id="${id}"]`);
    if (mark) {
      const parent = mark.parentNode;
      while (mark.firstChild) parent?.insertBefore(mark.firstChild, mark);
      parent?.removeChild(mark);
      parent?.normalize();
    }
    // Remove from storage
    const highlights = getHighlights(slug).filter(h => h.id !== id);
    saveHighlights(slug, highlights);
    setRemovePopover(null);
  }, [removePopover, slug]);

  return (
    <>
      {popover && (
        <div
          ref={popoverRef}
          className="highlight-popover"
          style={{ left: popover.x, top: popover.y }}
        >
          <button onClick={doHighlight}>Highlight</button>
          <button onClick={doShare}>Share</button>
        </div>
      )}
      {removePopover && (
        <div
          className="highlight-popover"
          style={{ left: removePopover.x, top: removePopover.y }}
        >
          <button onClick={doRemoveHighlight}>Remove</button>
        </div>
      )}
    </>
  );
}
