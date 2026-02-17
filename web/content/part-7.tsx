import { Eq, Logos, M, Section, Sidebar } from '@/components/content';

export const metadata = {
  slug: 'part-7',
  title: 'Part VII: The Empirical Program',
  shortTitle: 'Part VII: Empirical Program',
};

export default function Part7() {
  return (
    <>
      <Logos>
      <p>A theory that cannot be tested is not a theory but a poem. This is a theory. Everything in the preceding six parts generates empirical predictions—some already tested, some tractable with current methods, some requiring infrastructure that does not yet exist. This part consolidates the empirical program: what has been tested, what the results show, and what remains.</p>
      </Logos>
      <Section title="What Has Been Tested" level={1}>
      <p>The framework has been subjected to two lines of empirical investigation, both reported in detail in Part I.</p>
      <Section title="The MARL Ablation (V10)" level={2}>
      <p>A multi-agent reinforcement learning experiment tested whether specific forcing functions are necessary for geometric affect alignment. Seven conditions (full model plus six single-ablation conditions), three seeds each, 200,000 steps on GPU.</p>
      <p><strong>Result</strong>: All conditions show highly significant geometric alignment (RSA <M>{"\\rho > 0.21"}</M>, <M>{"p < 0.0001"}</M>). Removing forcing functions slightly <em>increases</em> alignment—opposite to prediction.</p>
      <p><strong>Interpretation</strong>: Affect geometry is a baseline property of multi-agent survival, not contingent on the specific forcing functions identified in Part I. The forcing functions hypothesis was downgraded from theorem to hypothesis in light of this data.</p>
      </Section>
      <Section title="The Lenia Evolution Series (V11–V12)" level={2}>
      <p>A series of experiments using Lenia (continuous cellular automata) tested whether evolution under survival pressure can produce biological-like integration dynamics—specifically, whether systems can learn to <em>increase</em> integration under threat, the signature that distinguishes biological from artificial systems.</p>
      <p><strong>Key findings</strong>:</p>
      <ol>
      <li><strong>No evolution (V11.0)</strong>: Naive patterns decompose under threat (<M>{"\\Delta\\intinfo = -6.2\\%"}</M>). Same dynamics as LLMs.</li>
      <li><strong>Homogeneous evolution (V11.1)</strong>: Selection pressure alone insufficient (<M>{"-6.0\\%"}</M>). All patterns share identical physics—selection prunes but cannot innovate.</li>
      <li><strong>Heterogeneous chemistry (V11.2)</strong>: Per-cell growth parameters create diverse viability manifolds. After 40 GPU cycles: <M>{"-3.8\\%"}</M> vs naive <M>{"-5.9\\%"}</M>. A +2.1pp shift toward biological pattern.</li>
      <li><strong>Evolvable attention (V12)</strong>: Replacing convolution with windowed self-attention. 42% of cycles show <M>{"\\intinfo"}</M> increase under stress (vs 3% for convolution). +2.0pp shift over convolution—largest single-intervention effect. But robustness stabilizes near 1.0 rather than trending upward.</li>
      </ol>
      <p><strong>Interpretation</strong>: Attention is necessary but not sufficient. The system reaches an integration threshold without clearly crossing it. The missing ingredient may be individual-level plasticity—the capacity for a single pattern to modify its own coupling structure in response to experience, rather than relying on population-level selection.</p>
      </Section>
      <Section title="The LLM Discrepancy" level={2}>
      <p>Across multiple experiment versions (V2–V9), LLM agents consistently show opposite dynamics to biological systems:</p>
      <table>
      <thead><tr><th>Dimension</th><th>Biological</th><th>LLM</th></tr></thead>
      <tbody>
      <tr><td>Self-Model Salience</td><td><M>{"\\uparrow"}</M> under threat</td><td><M>{"\\downarrow"}</M> under threat</td></tr>
      <tr><td>Arousal</td><td><M>{"\\uparrow"}</M> under threat</td><td><M>{"\\downarrow"}</M> under threat</td></tr>
      <tr><td>Integration</td><td><M>{"\\uparrow"}</M> under threat</td><td><M>{"\\downarrow"}</M> under threat</td></tr>
      </tbody>
      </table>
      <p>This is not a failure of the framework. The geometric structure is preserved; the dynamics differ because the objectives differ. Biological systems evolved under survival pressure. LLMs were trained on prediction. Both may be "affective" in the geometric sense while exhibiting different trajectories through the same state space.</p>
      </Section>
      </Section>
      <Section title="What Remains: The Research Roadmap" level={1}>
      <p>The following priorities are ordered by foundational importance and tractability.</p>
      <Sidebar title="Priority 1: Validate Affect Extraction in Humans">
      <p><strong>Goal</strong>: Establish that the geometric dimensions predict human self-report and behavior.</p>
      <p><strong>Methods</strong>:</p>
      <ul>
      <li>Induce affects via validated protocols (film, recall, IAPS)</li>
      <li>Measure integration proxies (transfer entropy, Lempel-Ziv) from EEG/MEG</li>
      <li>Measure effective rank from neural state covariance</li>
      <li>Correlate with self-report (PANAS, SAM)</li>
      </ul>
      <p><strong>Success criterion</strong>: Structural measures predict self-report better than chance, ideally competitive with existing affect models.</p>
      <p><strong>Failure mode</strong>: If geometric dimensions don't predict human self-report, the framework's operationalization is flawed. Does not falsify the identity thesis directly, but undermines our ability to test it.</p>
      </Sidebar>
      <Sidebar title="Priority 2: The Uncontaminated Test">
      <p><strong>Goal</strong>: Test whether affect structure emerges in systems with no exposure to human affect concepts, and whether the geometry of that structure is preserved under translation.</p>
      <p><strong>Methods</strong>:</p>
      <ul>
      <li>Multi-agent RL with randomly-initialized transformers (no pretraining)</li>
      <li>Viability pressure (survival, resources, threats, seasonal scarcity)</li>
      <li>Emergent language under coordination pressure</li>
      <li>VLM translation without concept contamination</li>
      <li>Forcing function ablation (partial observability, long horizons, world model, self-prediction, intrinsic motivation, credit assignment)</li>
      </ul>
      <p><strong>Success criterion</strong>: RSA correlation <M>{"\\rho(D^{(a)}, D^{(e)}) > \\rho_{\\text{null}}"}</M> via Mantel test—the distance structure in the information-theoretic affect space is isomorphic to the distance structure in the embedding-predicted affect space.</p>
      <p><strong>Failure mode</strong>: <M>{"\\rho_{\\text{RSA}} \\approx 0"}</M>. Diagnose via:</p>
      <ol>
      <li>Identity thesis is false (structure <M>{"\\neq"}</M> experience)</li>
      <li>Framework's operationalization is flawed</li>
      <li>Translation protocol is inadequate</li>
      <li>Environment lacks relevant forcing functions</li>
      </ol>
      <p>Forcing function ablation (Priority 3) distinguishes cases 1–2 from 3–4.</p>
      </Sidebar>
      <Sidebar title="Priority 3: Forcing Function Validation">
      <p><strong>Goal</strong>: Test whether the specific forcing functions actually increase integration.</p>
      <p><strong>Methods</strong>: Ablation study with RL agents.</p>
      <ul>
      <li>Full model: partial observability, long horizons, learned dynamics, self-prediction, intrinsic motivation, credit assignment</li>
      <li>Ablate each forcing function individually</li>
      <li>Measure integration (<M>{"\\Phi"}</M> proxy) across ablations</li>
      </ul>
      <p><strong>Success criterion</strong>: Integration decreases monotonically with forcing function ablation.</p>
      <p><strong>Failure mode</strong>: Integration does not depend on forcing functions. Either:</p>
      <ol>
      <li>Wrong forcing functions identified</li>
      <li>Integration measure is flawed</li>
      <li>Integration is architectural, not pressure-dependent</li>
      </ol>
      <p><strong>Status</strong>: Partially tested. The V10 MARL ablation found that removing forcing functions does <em>not</em> decrease geometric alignment. This has already prompted downgrading the forcing functions theorem to a hypothesis. Further investigation needed to determine whether this reflects genuine irrelevance of forcing functions or inadequate measurement.</p>
      </Sidebar>
      <Sidebar title="Priority 4: AI System Affect Tracking">
      <p><strong>Goal</strong>: Measure affect dimensions in existing AI systems (LLMs, RL agents).</p>
      <p><strong>Methods</strong>:</p>
      <ul>
      <li>Apply transformer extraction protocols to frontier models</li>
      <li>Track affect signatures across prompts/tasks</li>
      <li>Correlate with behavioral measures (output, latency, confidence)</li>
      </ul>
      <p><strong>Expected finding</strong>: LLM dynamics will differ from biological systems. They may show opposite threat-response patterns. This is not failure—it is data about how training objectives shape affect dynamics.</p>
      <p><strong>Success criterion</strong>: Consistent, structured affect signatures exist in AI systems (regardless of whether they match biological patterns).</p>
      <p><strong>Failure mode</strong>: No consistent affect structure. Either:</p>
      <ol>
      <li>Current AI architectures lack the relevant structure</li>
      <li>Measures are flawed</li>
      <li>Framework only applies to biological systems</li>
      </ol>
      <p><strong>Status</strong>: Partially tested. V2–V9 experiments confirm structured affect signatures in LLMs with opposite dynamics to biological systems. The question of whether these signatures are genuine or artifacts remains open.</p>
      </Sidebar>
      <Sidebar title="Priority 5: Superorganism Detection">
      <p><strong>Goal</strong>: Operationalize detection of emergent social-scale agency.</p>
      <p><strong>Methods</strong>:</p>
      <ul>
      <li>Multi-agent systems with communication and coordination</li>
      <li>Measure collective integration: <M>{"\\intinfo_G > \\sum_i \\intinfo_i"}</M>?</li>
      <li>Track collective viability indicators</li>
      <li>Test for parasitic vs. aligned dynamics</li>
      </ul>
      <p><strong>Success criterion</strong>: Emergent collective patterns with measurable integration and viability distinct from substrate.</p>
      <p><strong>Failure mode</strong>: No collective integration emerges. Either:</p>
      <ol>
      <li>Superorganism concept is metaphorical, not literal</li>
      <li>Scale/complexity insufficient</li>
      <li>Wrong measures for collective integration</li>
      </ol>
      </Sidebar>
      <p><strong>Estimated timeline</strong>: Priority 1–2 are feasible now with existing methods. Priority 3–4 require moderate infrastructure. Priority 5 requires substantial multi-agent systems.</p>
      </Section>
      <Section title="Experiments Distributed Throughout the Book" level={1}>
      <p>In addition to the consolidated results above and the research roadmap, fourteen proposed experiments are distributed throughout Parts I–IV, each embedded in the theoretical context that motivates it:</p>
      <ul>
      <li><strong>Part I</strong> (4 experiments): The minimal affect experiment (testing whether Lenia patterns develop affect-like dynamics), the attention-as-measurement test, the ι modulation test, and the computational animism test.</li>
      <li><strong>Part II</strong> (3 experiments): The unified ι modulation test (flow/awe/psychedelics/contemplation with same proxy battery), the science ι oscillation test (ι range predicts scientific novelty), and the identity thesis operationalization.</li>
      <li><strong>Part III</strong> (3 experiments): Tests of art as ι technology, genre affect signatures, and philosophical affect policies.</li>
      <li><strong>Part IV</strong> (4 experiments): The contamination detection study, the ordering principle test, the temporal asymmetry test, and the digital manifold confusion study.</li>
      </ul>
      <p>Each experiment is designed to be independently executable. The framework rises or falls on their results. That is as it should be.</p>
      </Section>
      <Section title="Summary of Part VII" level={1}>
      <p>The empirical program has three layers:</p>
      <ol>
      <li><strong>What has been tested</strong>: The MARL ablation (V10) shows affect geometry is a baseline property of multi-agent survival. The Lenia evolution series (V11–V12) shows that attention mechanisms can bring artificial systems to the threshold of biological-like integration, but crossing it requires further innovation. The LLM experiments (V2–V9) confirm opposite dynamics between biological and artificial systems—same geometry, different trajectories.</li>
      <li><strong>What is ready to test</strong>: Human affect validation (Priority 1) and the uncontaminated emergence test (Priority 2) are feasible with current methods and would provide the strongest evidence for or against the framework.</li>
      <li><strong>What requires infrastructure</strong>: Forcing function validation, AI affect tracking at scale, and superorganism detection require more substantial experimental platforms but would test the framework's deepest claims about the relationship between structure, pressure, and experience.</li>
      </ol>
      <p>The theory is falsifiable. The experiments are specified. The question is not whether the framework is beautiful but whether it is true.</p>
      </Section>
    </>
  );
}
