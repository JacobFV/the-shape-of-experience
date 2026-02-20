'use client';

import { type ComponentType } from 'react';
import PitchforkBifurcation from '../diagrams/PitchforkBifurcation';
import BenardCells from '../diagrams/BenardCells';
import ViabilityManifold from '../diagrams/ViabilityManifold';
import LipidBilayer from '../diagrams/LipidBilayer';
import ScaleLadder from '../diagrams/ScaleLadder';
import TripartiteAlignment from '../diagrams/TripartiteAlignment';
import ReductionChain from '../diagrams/ReductionChain';
import AffectSpace from '../diagrams/AffectSpace';
import ScaleInterventions from '../diagrams/ScaleInterventions';
import HistoricalTimeline from '../diagrams/HistoricalTimeline';
import EmergenceLadder from '../diagrams/EmergenceLadder';
import GradientCoupling from '../diagrams/GradientCoupling';
import BottleneckFurnace from '../diagrams/BottleneckFurnace';
import ProtocellArchitecture from '../diagrams/ProtocellArchitecture';
import NecessityChain from '../diagrams/NecessityChain';
import ProtocellGrid from '../diagrams/ProtocellGrid';
import IntegrationDistribution from '../diagrams/IntegrationDistribution';
import SubstrateLineage from '../diagrams/SubstrateLineage';
import VLMConvergence from '../diagrams/VLMConvergence';
import { FurnaceMechanism } from '../diagrams/FurnaceMechanism';
import IotaSpectrum from '../diagrams/IotaSpectrum';
import ContaminationDynamics from '../diagrams/ContaminationDynamics';
import SuperorganismTaxonomy from '../diagrams/SuperorganismTaxonomy';

/** Maps SVG paths to native React diagram components */
const NATIVE: Record<string, ComponentType> = {
  '/diagrams/part-1-0.svg': PitchforkBifurcation,
  '/diagrams/part-1-1.svg': BenardCells,
  '/diagrams/part-1-2.svg': ViabilityManifold,
  '/diagrams/part-1-3.svg': LipidBilayer,
  '/diagrams/part-1-4.svg': ScaleLadder,
  '/diagrams/part-1-5.svg': TripartiteAlignment,
  '/diagrams/part-2-0.svg': ReductionChain,
  '/diagrams/part-3-0.svg': AffectSpace,
  '/diagrams/part-4-0.svg': ScaleInterventions,
  '/diagrams/part-5-0.svg': HistoricalTimeline,
  '/diagrams/appendix-0.svg': EmergenceLadder,
  '/diagrams/appendix-1.svg': GradientCoupling,
  '/diagrams/appendix-2.svg': BottleneckFurnace,
  '/diagrams/appendix-3.svg': ProtocellArchitecture,
  '/diagrams/appendix-4.svg': NecessityChain,
  '/diagrams/appendix-5.svg': ProtocellGrid,
  '/diagrams/appendix-6.svg': IntegrationDistribution,
  '/diagrams/appendix-7.svg': SubstrateLineage,
  '/diagrams/appendix-8.svg': VLMConvergence,
  '/diagrams/appendix-9.svg': FurnaceMechanism,
  '/diagrams/part-2-1.svg': IotaSpectrum,
  '/diagrams/part-4-1.svg': ContaminationDynamics,
  '/diagrams/part-5-1.svg': SuperorganismTaxonomy,
};

interface DiagramProps {
  src: string;
  alt?: string;
}

export function Diagram({ src }: DiagramProps) {
  const Native = NATIVE[src];

  if (!Native) {
    throw new Error(`No native diagram component for: ${src}`);
  }

  return (
    <div className="center">
      <figure className="tikz-diagram">
        <Native />
      </figure>
    </div>
  );
}
