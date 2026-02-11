'use client';

import { useEffect, useRef, useCallback, useState } from 'react';
import { useAnnotations, Annotation } from '@/lib/hooks/useAnnotations';

function findNearestHeadingId(node: Node): string {
  let el: Element | null = node.nodeType === Node.TEXT_NODE ? node.parentElement : node as Element;
  while (el) {
    let prev: Element | null = el.previousElementSibling;
    while (prev) {
      if (/^H[1-3]$/i.test(prev.tagName) && prev.id) return prev.id;
      prev = prev.previousElementSibling;
    }
    el = el.parentElement;
  }
  return '';
}

function findNearestHeading(): { id: string; text: string } {
  const headings = document.querySelectorAll<HTMLElement>('.chapter-content h1[id], .chapter-content h2[id], .chapter-content h3[id]');
  const scrollY = window.scrollY + 100;
  let nearest = { id: '', text: 'Start of chapter' };
  for (const h of headings) {
    if (h.offsetTop <= scrollY) {
      nearest = { id: h.id, text: h.textContent?.trim() || '' };
    }
  }
  return nearest;
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

function findTextInContent(root: HTMLElement, highlight: { prefix: string; exact: string }): Range | null {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let node: Text | null;
  while ((node = walker.nextNode() as Text | null)) {
    const text = node.textContent || '';
    const idx = text.indexOf(highlight.exact);
    if (idx === -1) continue;

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
  const { items: annotations, add, update, remove, isAuth } = useAnnotations(slug);
  const [popover, setPopover] = useState<{ x: number; y: number; range: Range } | null>(null);
  const [editPopover, setEditPopover] = useState<{ x: number; y: number; annotation: Annotation } | null>(null);
  const [noteEditor, setNoteEditor] = useState<{ id: string; note: string } | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [audioAvailable, setAudioAvailable] = useState(false);

  // Check if audio is available on this page
  useEffect(() => {
    const check = () => setAudioAvailable(document.body.dataset.hasAudio === 'true');
    check();
    const observer = new MutationObserver(check);
    observer.observe(document.body, { attributes: true, attributeFilter: ['data-has-audio'] });
    return () => observer.disconnect();
  }, []);
  const popoverRef = useRef<HTMLDivElement>(null);
  const restoredRef = useRef(false);
  const prevAnnotationsRef = useRef<string>('');

  // Show toast briefly
  const showToast = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2000);
  }, []);

  // Restore highlights when annotations change
  useEffect(() => {
    const content = document.querySelector('.chapter-content');
    if (!content) return;

    const key = JSON.stringify(annotations.map((a) => a.id).sort());
    if (key === prevAnnotationsRef.current) return;
    prevAnnotationsRef.current = key;

    // Clear existing marks
    content.querySelectorAll('mark.user-highlight').forEach((mark) => {
      const parent = mark.parentNode;
      while (mark.firstChild) parent?.insertBefore(mark.firstChild, mark);
      parent?.removeChild(mark);
      parent?.normalize();
    });

    // Re-apply all annotations (skip bookmarks — they have no text to highlight)
    for (const a of annotations) {
      if (!a.exact) continue;
      const range = findTextInContent(content as HTMLElement, a);
      if (range) {
        const mark = document.createElement('mark');
        mark.className = `user-highlight${a.note ? ' has-note' : ''}`;
        mark.dataset.highlightId = a.id;
        if (a.isPublished) {
          mark.dataset.published = 'true';
        }
        try {
          range.surroundContents(mark);
        } catch { /* partial overlap */ }
      }
    }
  }, [annotations]);

  // Restore from URL ?hl= param on mount
  useEffect(() => {
    if (restoredRef.current) return;
    restoredRef.current = true;

    const params = new URLSearchParams(window.location.search);
    const hlParam = params.get('hl');
    if (!hlParam) return;

    const [prefix, exact, suffix] = hlParam.split('~').map(decodeURIComponent);
    if (!exact) return;

    setTimeout(() => {
      const content = document.querySelector('.chapter-content');
      if (!content) return;
      const range = findTextInContent(content as HTMLElement, { prefix: prefix || '', exact });
      if (range) {
        const mark = document.createElement('mark');
        mark.className = 'user-highlight';
        mark.style.animation = 'flash-highlight 2s ease';
        try {
          range.surroundContents(mark);
          mark.scrollIntoView({ behavior: 'smooth', block: 'center' });
          setTimeout(() => {
            const parent = mark.parentNode;
            while (mark.firstChild) parent?.insertBefore(mark.firstChild, mark);
            parent?.removeChild(mark);
            parent?.normalize();
          }, 3000);
        } catch { /* partial overlap */ }
      }
    }, 800);
  }, []);

  // Paragraph play button on click (no selection)
  const [paraPlayBtn, setParaPlayBtn] = useState<{ x: number; y: number; headingId: string } | null>(null);

  useEffect(() => {
    if (!paraPlayBtn) return;
    const timer = setTimeout(() => setParaPlayBtn(null), 3000);
    return () => clearTimeout(timer);
  }, [paraPlayBtn]);

  const onParagraphClick = useCallback((e: MouseEvent) => {
    if (!audioAvailable) return;
    const target = e.target as HTMLElement;
    // Only trigger on paragraph elements in chapter content
    const para = target.closest('.chapter-content p, .chapter-content li');
    if (!para) return;
    // Don't show if there's a text selection
    const sel = window.getSelection();
    if (sel && !sel.isCollapsed) return;
    // Don't show if clicking on interactive elements
    if (target.closest('button, a, mark, .highlight-popover, .para-play-btn')) return;

    const headingId = findNearestHeadingId(para);
    const rect = para.getBoundingClientRect();
    setParaPlayBtn({
      x: rect.left + rect.width / 2,
      y: rect.top - 8,
      headingId,
    });
  }, [audioAvailable]);

  useEffect(() => {
    document.addEventListener('click', onParagraphClick);
    return () => document.removeEventListener('click', onParagraphClick);
  }, [onParagraphClick]);

  const doParaPlay = useCallback(() => {
    if (!paraPlayBtn) return;
    window.dispatchEvent(new CustomEvent('play-section', { detail: { headingId: paraPlayBtn.headingId } }));
    setParaPlayBtn(null);
  }, [paraPlayBtn]);

  // Show popover for current selection
  const showSelectionPopover = useCallback(() => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.rangeCount) {
      setPopover(null);
      return;
    }

    const range = sel.getRangeAt(0);
    const text = range.toString().trim();
    if (text.length < 3) { setPopover(null); return; }

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

  // Handle text selection (desktop)
  const onMouseUp = useCallback((e: MouseEvent) => {
    const target = e.target as HTMLElement;
    if (target.closest('.highlight-popover') || target.closest('.reader-toolbar') || target.closest('.highlight-note-inline')) return;

    // Click on existing highlight
    if (target.closest('mark.user-highlight')) {
      const mark = target.closest('mark.user-highlight') as HTMLElement;
      const id = mark.dataset.highlightId;
      if (id) {
        const annotation = annotations.find((a) => a.id === id);
        if (annotation) {
          setEditPopover({ x: e.clientX, y: e.clientY - 40, annotation });
          setPopover(null);
          return;
        }
      }
    }

    setEditPopover(null);
    showSelectionPopover();
  }, [annotations, showSelectionPopover]);

  // Handle text selection (mobile) — selectionchange fires when iOS selection handles move
  const selectionChangeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const onSelectionChange = () => {
      if (selectionChangeTimer.current) clearTimeout(selectionChangeTimer.current);
      selectionChangeTimer.current = setTimeout(() => {
        // Only act on touch devices — mouseup handles desktop
        if (!('ontouchstart' in window)) return;
        const sel = window.getSelection();
        if (!sel || sel.isCollapsed) {
          setPopover(null);
          return;
        }
        showSelectionPopover();
      }, 300);
    };

    document.addEventListener('selectionchange', onSelectionChange);
    document.addEventListener('mouseup', onMouseUp);
    return () => {
      document.removeEventListener('selectionchange', onSelectionChange);
      document.removeEventListener('mouseup', onMouseUp);
      if (selectionChangeTimer.current) clearTimeout(selectionChangeTimer.current);
    };
  }, [onMouseUp, showSelectionPopover]);

  // Actions
  const doHighlight = useCallback(async () => {
    if (!popover) return;
    const ctx = getContext(popover.range);
    const headingId = findNearestHeadingId(popover.range.startContainer);
    await add({ slug, nearestHeadingId: headingId, nearestHeadingText: '', ...ctx, note: '' });
    window.getSelection()?.removeAllRanges();
    setPopover(null);
  }, [popover, slug, add]);

  const doNote = useCallback(async () => {
    if (!popover) return;
    const ctx = getContext(popover.range);
    const headingId = findNearestHeadingId(popover.range.startContainer);
    const annotation = await add({ slug, nearestHeadingId: headingId, nearestHeadingText: '', ...ctx, note: '' });
    window.getSelection()?.removeAllRanges();
    setPopover(null);
    setNoteEditor({ id: annotation.id, note: '' });
  }, [popover, slug, add]);

  const doBookmark = useCallback(async () => {
    if (!popover) return;
    const heading = findNearestHeading();
    await add({
      slug,
      nearestHeadingId: heading.id,
      nearestHeadingText: heading.text,
      prefix: '',
      exact: '',
      suffix: '',
      note: '',
    });
    window.getSelection()?.removeAllRanges();
    setPopover(null);
    showToast('Bookmarked');
  }, [popover, slug, add, showToast]);

  const doShare = useCallback(async () => {
    if (!popover) return;
    const ctx = getContext(popover.range);
    const hlParams = [ctx.prefix, ctx.exact, ctx.suffix].map(encodeURIComponent).join('~');
    // Build text fragment: #:~:text=[prefix-,]exact[,-suffix]
    const fragParts: string[] = [];
    if (ctx.prefix) fragParts.push(encodeURIComponent(ctx.prefix.slice(-20)) + '-,');
    fragParts.push(encodeURIComponent(ctx.exact));
    if (ctx.suffix) fragParts.push(',-' + encodeURIComponent(ctx.suffix.slice(0, 20)));
    const url = `${window.location.origin}/${slug}?hl=${hlParams}#:~:text=${fragParts.join('')}`;

    window.getSelection()?.removeAllRanges();
    setPopover(null);

    if (navigator.share) {
      try {
        await navigator.share({ url, title: document.title });
      } catch (e: unknown) {
        // User cancelled or share failed — copy as fallback
        if (e instanceof Error && e.name !== 'AbortError') {
          navigator.clipboard.writeText(url).catch(() => {});
          showToast('Link copied');
        }
      }
    } else {
      navigator.clipboard.writeText(url).catch(() => {});
      showToast('Link copied');
    }
  }, [popover, slug, showToast]);

  const doPlay = useCallback(() => {
    if (!popover) return;
    const headingId = findNearestHeadingId(popover.range.startContainer);
    window.dispatchEvent(new CustomEvent('play-section', { detail: { headingId } }));
    window.getSelection()?.removeAllRanges();
    setPopover(null);
  }, [popover]);

  const doEditNote = useCallback(() => {
    if (!editPopover) return;
    setNoteEditor({ id: editPopover.annotation.id, note: editPopover.annotation.note });
    setEditPopover(null);
  }, [editPopover]);

  const doTogglePublish = useCallback(async () => {
    if (!editPopover || !isAuth) return;
    const a = editPopover.annotation;
    await update(a.id, { isPublished: !a.isPublished });
    setEditPopover(null);
    showToast(a.isPublished ? 'Unpublished' : 'Published');
  }, [editPopover, isAuth, update, showToast]);

  const doRemove = useCallback(async () => {
    if (!editPopover) return;
    await remove(editPopover.annotation.id);
    setEditPopover(null);
  }, [editPopover, remove]);

  const saveNote = useCallback(async () => {
    if (!noteEditor) return;
    await update(noteEditor.id, { note: noteEditor.note });
    setNoteEditor(null);
  }, [noteEditor, update]);

  return (
    <>
      {/* Selection popover */}
      {popover && (
        <div
          ref={popoverRef}
          className="highlight-popover"
          style={{ left: popover.x, top: popover.y }}
        >
          <button onClick={doHighlight}>Highlight</button>
          <button onClick={doNote}>Note</button>
          <button onClick={doBookmark}>Bookmark</button>
          <button onClick={doShare}>Share</button>
          {audioAvailable && <button onClick={doPlay}>Play</button>}
        </div>
      )}

      {/* Existing highlight popover */}
      {editPopover && (
        <div
          className="highlight-popover"
          style={{ left: editPopover.x, top: editPopover.y }}
        >
          <button onClick={doEditNote}>
            {editPopover.annotation.note ? 'Edit Note' : 'Add Note'}
          </button>
          {isAuth && (
            <button onClick={doTogglePublish}>
              {editPopover.annotation.isPublished ? 'Unpublish' : 'Publish'}
            </button>
          )}
          <button onClick={doRemove}>Remove</button>
        </div>
      )}

      {/* Inline note editor */}
      {noteEditor && (
        <div className="highlight-note-inline" style={{ maxWidth: 'var(--content-width)' }}>
          <textarea
            className="highlight-note-textarea"
            value={noteEditor.note}
            onChange={(e) => setNoteEditor({ ...noteEditor, note: e.target.value })}
            placeholder="Write your note..."
            autoFocus
          />
          <div className="highlight-note-actions">
            <button onClick={saveNote}>Save</button>
            <button onClick={() => setNoteEditor(null)}>Cancel</button>
          </div>
        </div>
      )}

      {/* Paragraph play button */}
      {paraPlayBtn && (
        <div
          className="para-play-btn"
          style={{ left: paraPlayBtn.x, top: paraPlayBtn.y }}
        >
          <button onClick={doParaPlay} aria-label="Play from here">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
              <path d="M8 5v14l11-7z" />
            </svg>
            Play
          </button>
        </div>
      )}

      {/* Toast */}
      {toast && <div className="toast">{toast}</div>}
    </>
  );
}
