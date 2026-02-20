'use client';

import dynamic from 'next/dynamic';

const ExperimentMap = dynamic(() => import('./ExperimentMap'), { ssr: false });

export default function ExperimentMapWrapper() {
  return <ExperimentMap />;
}
