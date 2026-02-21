import React from "react";
import { Composition } from "remotion";
import { BottleneckFurnaceVideo } from "./BottleneckFurnace";
import { IntegrationTrajectoryVideo } from "./IntegrationTrajectory";
import { GradientCouplingVideo } from "./GradientCoupling";
import { LLMAffectContrastVideo } from "./LLMAffectContrast";
import { WallBreakingVideo } from "./WallBreaking";
import { FalsificationScoreboardVideo } from "./FalsificationScoreboard";
import { LanguageEmergenceVideo } from "./LanguageEmergence";
import { GeometryIsCheapVideo } from "./GeometryIsCheap";
import { AffectMotifsVideo } from "./AffectMotifs";
import { DevelopmentalOrderingVideo } from "./DevelopmentalOrdering";
import { IdentificationScopeVideo } from "./IdentificationScope";
import { IotaHistoricalVideo } from "./IotaHistorical";
import { SuperorganismLifecycleVideo } from "./SuperorganismLifecycle";
import { AffectTechnologyVideo } from "./AffectTechnology";
import { AxialTransitionVideo } from "./AxialTransition";
import { NecessityArgumentVideo } from "./NecessityArgument";
import type { ThemeMode } from "./themes";

const COMPOSITIONS: {
  id: string;
  component: React.FC<{ theme?: ThemeMode }>;
  durationInFrames: number;
}[] = [
  { id: "BottleneckFurnace", component: BottleneckFurnaceVideo, durationInFrames: 360 },
  { id: "IntegrationTrajectory", component: IntegrationTrajectoryVideo, durationInFrames: 300 },
  { id: "GradientCoupling", component: GradientCouplingVideo, durationInFrames: 300 },
  { id: "LLMAffectContrast", component: LLMAffectContrastVideo, durationInFrames: 240 },
  { id: "WallBreaking", component: WallBreakingVideo, durationInFrames: 300 },
  { id: "FalsificationScoreboard", component: FalsificationScoreboardVideo, durationInFrames: 360 },
  { id: "LanguageEmergence", component: LanguageEmergenceVideo, durationInFrames: 300 },
  { id: "GeometryIsCheap", component: GeometryIsCheapVideo, durationInFrames: 300 },
  { id: "AffectMotifs", component: AffectMotifsVideo, durationInFrames: 360 },
  { id: "DevelopmentalOrdering", component: DevelopmentalOrderingVideo, durationInFrames: 360 },
  { id: "IdentificationScope", component: IdentificationScopeVideo, durationInFrames: 360 },
  { id: "IotaHistorical", component: IotaHistoricalVideo, durationInFrames: 360 },
  { id: "SuperorganismLifecycle", component: SuperorganismLifecycleVideo, durationInFrames: 360 },
  { id: "AffectTechnology", component: AffectTechnologyVideo, durationInFrames: 360 },
  { id: "AxialTransition", component: AxialTransitionVideo, durationInFrames: 360 },
  { id: "NecessityArgument", component: NecessityArgumentVideo, durationInFrames: 300 },
];

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {COMPOSITIONS.map(({ id, component, durationInFrames }) => (
        <React.Fragment key={id}>
          <Composition
            id={id}
            component={component}
            durationInFrames={durationInFrames}
            fps={30}
            width={1080}
            height={720}
            defaultProps={{ theme: "dark" as const }}
          />
          <Composition
            id={`${id}Light`}
            component={component}
            durationInFrames={durationInFrames}
            fps={30}
            width={1080}
            height={720}
            defaultProps={{ theme: "light" as const }}
          />
        </React.Fragment>
      ))}
    </>
  );
};
