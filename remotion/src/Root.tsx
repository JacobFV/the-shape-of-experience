import { Composition } from "remotion";
import { BottleneckFurnaceVideo } from "./BottleneckFurnace";
import { IntegrationTrajectoryVideo } from "./IntegrationTrajectory";
import { GradientCouplingVideo } from "./GradientCoupling";

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
    </>
  );
};
