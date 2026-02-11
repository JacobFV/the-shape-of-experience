'use client';

import { useSession } from 'next-auth/react';
import { usePathname } from 'next/navigation';
import { useEffect } from 'react';
import TableOfContents from '../components/TableOfContents';
import { useMobileUI } from '../lib/MobileUIContext';

interface Section {
  level: number;
  id: string;
  text: string;
}

export default function Sidebar({ sectionData }: { sectionData?: Record<string, Section[]> }) {
  const { sidebarOpen, setSidebarOpen } = useMobileUI();
  const { status } = useSession();
  const pathname = usePathname();

  useEffect(() => {
    setSidebarOpen(false);
  }, [pathname, setSidebarOpen]);

  return (
    <>
      <aside className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <TableOfContents onNavigate={() => setSidebarOpen(false)} sectionData={sectionData} />
        {status === 'authenticated' && (
          <div className="toc-pdf" style={{ marginTop: '1rem', paddingTop: '0.75rem' }}>
            <ul className="toc" style={{ listStyle: 'none', padding: 0 }}>
              <li>
                <a href="/library" onClick={() => setSidebarOpen(false)} className="toc-link" style={{ fontFamily: 'var(--font-sans)', fontSize: '0.82rem', color: 'var(--text-secondary)', textDecoration: 'none', display: 'block', padding: '0.25rem 0.5rem', borderRadius: '4px' }}>
                  Library
                </a>
              </li>
              <li>
                <a href="/settings" onClick={() => setSidebarOpen(false)} className="toc-link" style={{ fontFamily: 'var(--font-sans)', fontSize: '0.82rem', color: 'var(--text-secondary)', textDecoration: 'none', display: 'block', padding: '0.25rem 0.5rem', borderRadius: '4px' }}>
                  Settings
                </a>
              </li>
            </ul>
          </div>
        )}
      </aside>
      {sidebarOpen && <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />}
    </>
  );
}
