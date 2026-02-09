'use client';

import { useState } from 'react';
import TableOfContents from '../components/TableOfContents';

interface Section {
  level: number;
  id: string;
  text: string;
}

export default function Sidebar({ sectionData }: { sectionData?: Record<string, Section[]> }) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        className="sidebar-toggle"
        onClick={() => setOpen(!open)}
        aria-label="Toggle table of contents"
      >
        {open ? '\u2715' : '\u2630'}
      </button>
      <aside className={`sidebar ${open ? 'sidebar-open' : ''}`}>
        <TableOfContents onNavigate={() => setOpen(false)} sectionData={sectionData} />
      </aside>
      {open && <div className="sidebar-overlay" onClick={() => setOpen(false)} />}
    </>
  );
}
