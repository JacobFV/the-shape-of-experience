'use client';

import dynamic from 'next/dynamic';

const AffectTrajectory3D = dynamic(
  () => import('@/components/visualizations/AffectTrajectory3D'),
  {
    ssr: false,
    loading: () => (
      <div
        style={{
          height: 500,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#666',
          fontFamily: 'monospace',
          fontSize: '0.85rem',
        }}
      >
        Loading 3D visualization...
      </div>
    ),
  }
);

export default function AffectTrajectory3DWrapper({
  demoMode = true,
  trajectoryUrl,
}: {
  demoMode?: boolean;
  trajectoryUrl?: string;
}) {
  return <AffectTrajectory3D demoMode={demoMode} trajectoryUrl={trajectoryUrl} />;
}
