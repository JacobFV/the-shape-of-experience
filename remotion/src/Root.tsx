import { Composition } from "remotion";
import { BottleneckFurnaceVideo } from "./BottleneckFurnace";
import { IntegrationTrajectoryVideo } from "./IntegrationTrajectory";
import { GradientCouplingVideo } from "./GradientCoupling";
import { LLMAffectContrastVideo } from "./LLMAffectContrast";
import { WallBreakingVideo } from "./WallBreaking";
import { FalsificationScoreboardVideo } from "./FalsificationScoreboard";
import { LanguageEmergenceVideo } from "./LanguageEmergence";
import { GeometryIsCheapVideo } from "./GeometryIsCheap";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="BottleneckFurnace"
        component={BottleneckFurnaceVideo}
        durationInFrames={360}
        fps={30}
        width={1080}
        height={720}
      />
      <Composition
        id="IntegrationTrajectory"
        component={IntegrationTrajectoryVideo}
        durationInFrames={300}
        fps={30}
        width={1080}
        height={720}
      />
      <Composition
        id="GradientCoupling"
        component={GradientCouplingVideo}
        durationInFrames={300}
        fps={30}
        width={1080}
        height={720}
      />
      <Composition
        id="LLMAffectContrast"
        component={LLMAffectContrastVideo}
        durationInFrames={240}
        fps={30}
        width={1080}
        height={720}
      />
      <Composition
        id="WallBreaking"
        component={WallBreakingVideo}
        durationInFrames={300}
        fps={30}
        width={1080}
        height={720}
      />
      <Composition
        id="FalsificationScoreboard"
        component={FalsificationScoreboardVideo}
        durationInFrames={360}
        fps={30}
        width={1080}
        height={720}
      />
      <Composition
        id="LanguageEmergence"
        component={LanguageEmergenceVideo}
        durationInFrames={300}
        fps={30}
        width={1080}
        height={720}
      />
      <Composition
        id="GeometryIsCheap"
        component={GeometryIsCheapVideo}
        durationInFrames={300}
        fps={30}
        width={1080}
        height={720}
      />
    </>
  );
};
