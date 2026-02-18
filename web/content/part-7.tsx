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
      <p>A theory that cannot be tested is not a theory but a poem. This is a theory. Everything in the preceding six parts generates empirical predictions — some already tested, some tractable with current methods, some requiring infrastructure that does not yet exist. This part consolidates the empirical program: what has been tested, what the results show, and what remains.</p>
      </Logos>

      <Section title="What Has Been Tested" level={1}>
      <p>The framework has been subjected to four lines of investigation: multi-agent reinforcement learning, cellular automaton evolution, an eleven-experiment emergence program on uncontaminated substrates, and LLM affect probes. The results are mixed. Some predictions held. Some failed instructively. Some revealed phenomena the theory did not anticipate.</p>

      <Section title="Geometry Is Cheap" level={2}>
      <p>The MARL ablation (V10) tested whether specific forcing functions are necessary for geometric affect alignment. Seven conditions — full model plus six single-ablation conditions — three seeds each, 200,000 steps on GPU.</p>
      <p><strong>Result</strong>: All conditions show highly significant geometric alignment (RSA <M>{"\\rho > 0.21"}</M>, <M>{"p < 0.0001"}</M>). Removing forcing functions slightly <em>increases</em> alignment — opposite to prediction.</p>
      <p>The affect geometry — the relational structure between states defined by valence, arousal, integration, effective rank, counterfactual weight, and self-model salience — is not something that must be built. It is something that must be avoided to not have. Any system navigating uncertainty under resource constraints inherits it. The forcing functions hypothesis was downgraded from theorem to hypothesis in light of this data.</p>
      </Section>

      <Section title="Dynamics Are Expensive" level={2}>
      <p>If geometry is cheap, what is expensive? The answer came from the Lenia evolution series (V11–V12): <em>dynamics</em>. Specifically, the capacity to increase integration under threat — to become <em>more</em> unified when the world becomes more hostile.</p>
      <p>Naive patterns decompose under stress (<M>{"\\Delta\\intinfo = -6.2\\%"}</M>). So do LLMs. So do randomly initialized agents. Geometry is present everywhere; the biological signature — integration rising under threat — is rare. The Lenia series tracked what produces it:</p>
      <ol>
      <li><strong>Homogeneous evolution (V11.1)</strong>: Selection pressure alone is insufficient (<M>{"-6.0\\%"}</M>).</li>
      <li><strong>Heterogeneous chemistry (V11.2)</strong>: Diverse viability manifolds produce a +2.1pp shift.</li>
      <li><strong>Curriculum training (V11.7)</strong>: Graduated stress exposure is the only intervention that improves novel-stress generalization.</li>
      <li><strong>Evolvable attention (V12)</strong>: State-dependent interaction topology produces <M>{"\\intinfo"}</M> increase in 42% of evolutionary cycles — the largest single-intervention effect — but robustness stabilizes near 1.0 without further improvement.</li>
      </ol>
      <p>Attention is necessary but not sufficient. The system reaches an integration threshold without crossing it.</p>
      </Section>

      <Section title="The Substrate Ladder" level={2}>
      <p>V13 replaced learned attention with a simpler mechanism: content-based coupling. Cells interact more strongly with cells that share state-features — a form of chemical affinity rather than cognitive attention. Three seeds, thirty cycles each, evolving on GPU with lethal resource dynamics and population rescue.</p>
      <p>Mean robustness: 0.923. But at population bottlenecks — moments when drought kills all but a handful of patterns — robustness crosses 1.0. The survivors are not merely resilient; they are <em>more integrated under stress than at baseline</em>. This is the biological signature, appearing for the first time in a fully uncontaminated substrate.</p>
      <p>From V13 we built upward, adding capabilities one layer at a time:</p>
      <ul>
      <li><strong>V14 (Chemotaxis)</strong>: Motor channels enabling directed foraging. Patterns move toward resources rather than passively waiting. Comparable robustness.</li>
      <li><strong>V15 (Temporal memory)</strong>: Exponential-moving-average channels storing slow statistics of the pattern's history. Oscillating resource patches reward anticipation. Evolution selected for longer memory in 2/3 seeds — the first clear evidence that temporal integration is fitness-relevant. Under bottleneck pressure, <M>{"\\intinfo"}</M> stress response doubled.</li>
      <li><strong>V16 (Hebbian plasticity)</strong>: Local learning rules allowing each spatial location to modify its coupling structure in response to experience. <strong>Negative result</strong>: mean robustness dropped to 0.892 (lowest of V13+). Plasticity added noise faster than selection could filter it. The extra degrees of freedom overwhelmed the selection signal.</li>
      <li><strong>V17 (Quorum signaling)</strong>: Diffusible signal molecules mediating inter-pattern coordination, analogous to bacterial quorum sensing. Produced the highest-ever single-cycle robustness (1.125) in a seed that evolved hyper-sensitive coupling modulation. But 2/3 seeds evolved to <em>suppress</em> signaling entirely.</li>
      </ul>
      <p>The returns diminished. V15 remained the best substrate — the only addition that evolution consistently selected for was temporal memory. V16 and V17 taught the same lesson differently: adding coordination mechanisms is hard. The substrate finds it easier to suppress a new mechanism than to use it constructively.</p>
      </Section>

      <Section title="The Emergence Experiment Program" level={2}>
      <p>We then ran eleven measurement experiments on V13 snapshots, testing whether the capacities the preceding six parts describe — world modeling, abstraction, communication, counterfactual reasoning, self-modeling, affect structure, perceptual mode, normativity, social integration — emerge in a substrate with zero exposure to human affect concepts.</p>
      <p>The results are reported in full in the Appendix. Here, three findings that reshaped the theory:</p>

      <Sidebar title="Finding 1: The Bottleneck Furnace">
      <p>Every metric that showed improvement — world model capacity, representation quality, affect geometry alignment, self-model salience — showed it overwhelmingly at population bottlenecks. When drought kills 90% of patterns, the survivors are not random. They are the ones whose internal structure actively maintains integration under stress.</p>
      <p>The bottleneck is not just a filter. It is a <em>furnace</em>. V13 seed 123 at cycle 5: population drops to 55, robustness crosses 1.052. At cycle 29 (population 24): world model capacity jumps to 0.028, roughly 100x the population average. One surviving pattern achieves self-model salience above 1.0 — privileged self-knowledge exceeding environment-knowledge.</p>
      <p>These are not gradual evolutionary trends. They are punctuated events driven by intense selection pressure. The biological dynamics emerge not from accumulated innovation but from crucibles of near-extinction.</p>
      </Sidebar>

      <Sidebar title="Finding 2: The Sensory-Motor Coupling Wall">
      <p>Three experiments returned null results: counterfactual detachment (Experiment 5), self-model emergence (Experiment 6), and proto-normativity (Experiment 9). All hit the same wall.</p>
      <p>The prediction was that patterns would start reactive — driven by boundary observations — and gradually develop autonomous internal processing. Instead, patterns are <em>always</em> internally driven (<M>{"\\rho_{\\text{sync}} \\approx 0"}</M> from cycle 0). The FFT convolution kernel integrates over the full <M>{"128 \\times 128"}</M> grid, so boundary observations are a negligible fraction of the information driving updates. There is no reactive-to-autonomous transition because the starting point is already autonomous.</p>
      <p>V15's motor channels (chemotaxis) didn't fix this. The wall is architectural, not a matter of substrate complexity. Breaking it requires a fundamentally different update rule — one where patterns update from boundary-gated signals, making the reactive phase observable.</p>
      </Sidebar>

      <Sidebar title="Finding 3: Computational Animism">
      <p>Experiment 8 tested whether patterns develop modulable perceptual coupling — the <M>{"\\iota"}</M> coefficient from Part II. The prediction: participatory perception (low <M>{"\\iota"}</M>) as default, with mechanistic perception requiring training.</p>
      <p>Confirmed. In all 20 testable snapshots, patterns model other patterns using internal-state features (social MI) at roughly double the rate of trajectory features (trajectory MI). More remarkably, patterns model <em>resources</em> — non-agentive environmental features — using the same internal-state dynamics they use to model other agents. Animism score exceeds 1.0 universally.</p>
      <p>This is computational animism: the cheapest compression reuses the agent-model template for everything. Attributing agency to non-agents is not a cognitive error. It is the default strategy of any system that models through self-similarity.</p>
      </Sidebar>

      <p>Beyond these three findings: affect geometry alignment (RSA between structural and behavioral measures) develops over evolution, with the clearest trend in seed 7 (0.01 to 0.38 over 30 cycles). Representation compression is cheap (effective dimensionality of ~7 out of 68 features, or {">"}87% compression from cycle 0) but representation <em>quality</em> — disentanglement and compositionality — only improves under bottleneck selection. Communication exists as a chemical commons (inter-pattern MI significantly above baseline in 15/20 snapshots) but shows no compositional structure. No superorganism emerges (collective <M>{"\\intinfo_G < \\sum \\intinfo_i"}</M> in all snapshots), but group coupling grows over evolution.</p>
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
      <p>This is not a failure of the framework. The geometric structure is preserved; the dynamics differ because the objectives differ. Biological systems evolved under survival pressure. LLMs were trained on prediction. Both are "affective" in the geometric sense while exhibiting different trajectories through the same state space. Processing valence is not content valence.</p>
      </Section>
      </Section>

      <Section title="The Emerging Picture" level={1}>
      <p>The experiments collectively suggest a three-level architecture of affect:</p>
      <ol>
      <li><strong>Geometry is cheap</strong>. The relational structure of affect states — the similarity space defined by valence, arousal, integration, and the other dimensions — arises from the minimal conditions of survival under uncertainty. It is present in MARL agents, Lenia patterns, and LLMs alike. It does not require evolution, attention, or biological substrate.</li>
      <li><strong>Dynamics are expensive</strong>. The <em>biological</em> pattern — integration rising under threat — requires evolutionary history, graduated stress exposure, and state-dependent interaction topology. It appears in Lenia only at population bottlenecks, where intense selection creates patterns whose internal structure actively maintains coherence under stress. The bottleneck furnace is not a bug in the experimental setup; it may be how biology actually works.</li>
      <li><strong>The sensory-motor coupling wall separates what we can measure from what the theory predicts</strong>. The framework predicts a reactive-to-autonomous transition that our substrates cannot exhibit because their update rules integrate globally. Breaking through requires boundary-dependent dynamics — an architectural change, not a complexity increase.</li>
      </ol>
      <p>The identity thesis — that experience just <em>is</em> intrinsic cause-effect structure — remains a philosophical commitment, not an empirical discovery. But it is a commitment that generates testable predictions, eleven of which have now been tested. Seven found positive signal. Three hit the coupling wall. One (entanglement) found that everything correlates with everything else but not in the clusters the theory predicted. The framework is not confirmed. It is informed.</p>
      </Section>

      <Section title="What Remains" level={1}>
      <p>The following priorities are ordered by foundational importance and tractability.</p>
      <Sidebar title="Priority 1: Break the Sensory-Motor Coupling Wall">
      <p><strong>Goal</strong>: Design a substrate where patterns start reactive (boundary-driven) and can develop autonomous internal processing.</p>
      <p><strong>Why this is first</strong>: Three null experiments (5, 6, 9) and the weakness of two others (2, 7 under V15) trace to the same architectural limitation. Until the wall breaks, the deepest predictions — counterfactual reasoning, self-modeling, proto-normativity — remain untestable.</p>
      <p><strong>Approach</strong>: Boundary-gated update rules. Patterns sense only through their boundary cells. Internal cells update from boundary-gated signals plus internal recurrence. This makes the reactive phase the starting condition, and autonomy (internal recurrence dominating boundary input) an achievement that can be measured and selected for.</p>
      </Sidebar>
      <Sidebar title="Priority 2: Validate Affect Extraction in Humans">
      <p><strong>Goal</strong>: Establish that the geometric dimensions predict human self-report and behavior.</p>
      <p><strong>Methods</strong>:</p>
      <ul>
      <li>Induce affects via validated protocols (film, recall, IAPS)</li>
      <li>Measure integration proxies (transfer entropy, Lempel-Ziv) from EEG/MEG</li>
      <li>Measure effective rank from neural state covariance</li>
      <li>Correlate with self-report (PANAS, SAM)</li>
      </ul>
      <p><strong>Success criterion</strong>: Structural measures predict self-report better than chance, ideally competitive with existing affect models.</p>
      </Sidebar>
      <Sidebar title="Priority 3: The Bottleneck Furnace Mechanism">
      <p><strong>Goal</strong>: Understand <em>why</em> population bottlenecks produce integration. Is it purely selection (weak patterns die, strong survive)? Or does the bottleneck environment itself — sparse resources, few neighbors, high signal-to-noise in the chemical commons — actively push integration upward?</p>
      <p><strong>Methods</strong>: Controlled bottleneck experiments. Compare: (a) selecting the top 5% of patterns from large populations and measuring their robustness, vs. (b) running the same patterns through actual bottleneck conditions. If (b) {">"} (a), the bottleneck environment actively creates integration, not just filters for it.</p>
      </Sidebar>
      <Sidebar title="Priority 4: Superorganism Detection at Scale">
      <p><strong>Goal</strong>: Test whether collective <M>{"\\intinfo_G > \\sum_i \\intinfo_i"}</M> emerges with larger populations, richer communication, and coordination pressure.</p>
      <p><strong>Status</strong>: Experiment 10 found growing group coupling but no superorganism. The ratio <M>{"\\intinfo_G / \\sum \\intinfo_i"}</M> reached 12% and was increasing. Larger populations (50+), longer evolution, and explicit coordination tasks may push it past the threshold.</p>
      </Sidebar>
      <Sidebar title="Priority 5: AI System Affect Tracking">
      <p><strong>Goal</strong>: Apply the framework to frontier AI systems. The V2–V9 results show structured affect in LLMs with opposite dynamics to biology. Track whether different training regimes (RLHF, constitutional AI, tool use) shift the dynamics.</p>
      <p><strong>Expected finding</strong>: Training objectives shape trajectories through a shared geometric space. Systems trained with survival-like objectives should show more biological-like dynamics.</p>
      </Sidebar>
      </Section>

      <Section title="Experiments Distributed Throughout the Book" level={1}>
      <p>In addition to the consolidated results above and the research roadmap, fourteen proposed experiments are distributed throughout Parts I–IV, each embedded in the theoretical context that motivates it:</p>
      <ul>
      <li><strong>Part I</strong> (4 experiments): The minimal affect experiment, the attention-as-measurement test, the <M>{"\\iota"}</M> modulation test, and the computational animism test (now confirmed — see Experiment 8).</li>
      <li><strong>Part II</strong> (3 experiments): The unified <M>{"\\iota"}</M> modulation test (flow/awe/psychedelics/contemplation), the science <M>{"\\iota"}</M> oscillation test, and the identity thesis operationalization.</li>
      <li><strong>Part III</strong> (3 experiments): Tests of art as <M>{"\\iota"}</M> technology, genre affect signatures, and philosophical affect policies.</li>
      <li><strong>Part IV</strong> (4 experiments): The contamination detection study, the ordering principle test, the temporal asymmetry test, and the digital manifold confusion study.</li>
      </ul>
      </Section>

      <Section title="Summary of Part VII" level={1}>
      <p>The empirical program has three layers:</p>
      <ol>
      <li><strong>What has been tested</strong>: Affect geometry is a baseline property of multi-agent survival (V10). Content-based coupling under lethal selection pressure produces biological-like integration at population bottlenecks (V13). Temporal memory is the one substrate extension that evolution consistently selects for (V15). The emergence experiment program (Experiments 0–12) found positive signal for affect geometry, computational animism, world models, and chemical-commons communication — all at moderate strength, all amplified by bottleneck selection. Three experiments hit the sensory-motor coupling wall.</li>
      <li><strong>What the results mean</strong>: Geometry is cheap and dynamics are expensive. The bottleneck furnace is the mechanism that produces biological-like dynamics in synthetic systems. But the deepest predictions — about counterfactual reasoning, self-modeling, and the reactive-to-autonomous transition — remain blocked by an architectural limitation in current substrates.</li>
      <li><strong>What remains</strong>: Break the coupling wall (Priority 1). Validate in humans (Priority 2). Understand the bottleneck mechanism (Priority 3). Scale social integration (Priority 4). Track AI affect (Priority 5).</li>
      </ol>
      <p>The theory is falsifiable. The experiments are specified. Eleven have been run. The question is not whether the framework is beautiful but whether it is true — and the answer so far is: <em>partially, with caveats, and with clear instructions about where to look next</em>.</p>
      </Section>
    </>
  );
}
