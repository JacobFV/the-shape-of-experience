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
      <p>A theory that cannot be tested is not a theory but a poem. This is a theory. Everything in the preceding six parts generates empirical predictions — some already tested, some tractable with current methods, some requiring infrastructure that does not yet exist. This part consolidates the empirical program: what has been tested, what the results show, what they mean for the bridge between physics and psychology, and what remains.</p>
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
      <li><strong>V16 (Hebbian plasticity)</strong>: <strong>Negative result</strong>. Mean robustness dropped to 0.892 (lowest of V13+). Plasticity added noise faster than selection could filter it.</li>
      <li><strong>V17 (Quorum signaling)</strong>: Highest-ever single-cycle robustness (1.125). But 2/3 seeds evolved to <em>suppress</em> signaling entirely.</li>
      <li><strong>V18 (Boundary-dependent dynamics)</strong>: An insulation field computed from pattern morphology creates distinct boundary and interior signal domains. Boundary cells receive external convolution; interior cells receive only local recurrence. Three seeds evolved three different membrane strategies — permeable, thick-insulated, and filamentous. Mean robustness: 0.969, the highest of any substrate. Peak: 1.651. But <em>internal gain evolved down in all three seeds</em>. Evolution preferred thin, porous membranes over thick insulated cores.</li>
      </ul>
      <p>The substrate ladder taught two lessons. First: the only addition evolution consistently selected for was temporal memory. Plasticity, signaling, and boundary complexity were either suppressed or reduced. Second: raw robustness kept climbing (V13: 0.923, V15: 0.907, V18: 0.969), but this did not translate into richer cognitive dynamics. Making patterns more resilient is not the same as making them more minded.</p>
      </Section>

      <Section title="The Emergence Experiment Program" level={2}>
      <p>We then ran eleven measurement experiments on V13 snapshots, testing whether the capacities the preceding six parts describe — world modeling, abstraction, communication, counterfactual reasoning, self-modeling, affect structure, perceptual mode, normativity, social integration — emerge in a substrate with zero exposure to human affect concepts. Key experiments were re-run on V15 and V18 substrates.</p>
      <p>The results are reported in full in the Appendix. Here, three findings that reshaped the theory:</p>

      <Sidebar title="Finding 1: The Bottleneck Furnace">
      <p>Every metric that showed improvement — world model capacity, representation quality, affect geometry alignment, self-model salience — showed it overwhelmingly at population bottlenecks. When drought kills 90% of patterns, the survivors are not random. They are the ones whose internal structure actively maintains integration under stress.</p>
      <p>The bottleneck is not just a filter. It is a <em>furnace</em>. V13 seed 123 at cycle 5: population drops to 55, robustness crosses 1.052. At cycle 29 (population 24): world model capacity jumps to 0.028, roughly 100x the population average. One surviving pattern achieves self-model salience above 1.0 — privileged self-knowledge exceeding environment-knowledge.</p>
      <p>These are not gradual evolutionary trends. They are punctuated events driven by intense selection pressure. The biological dynamics emerge not from accumulated innovation but from crucibles of near-extinction.</p>
      </Sidebar>

      <Sidebar title="Finding 2: The Sensory-Motor Coupling Wall">
      <p>Three experiments returned null results: counterfactual detachment (Experiment 5), self-model emergence (Experiment 6), and proto-normativity (Experiment 9). All hit the same wall.</p>
      <p>The prediction was that patterns would start reactive — driven by boundary observations — and gradually develop autonomous internal processing. Instead, patterns are <em>always</em> internally driven (<M>{"\\rho_{\\text{sync}} \\approx 0"}</M> from cycle 0). There is no reactive-to-autonomous transition because the starting point is already autonomous.</p>
      <p>We attempted to break this wall directly. V15 added motor channels — chemotaxis, directed motion. No change. V18 introduced an insulation field creating genuine boundary and interior signal domains, with distinct external and internal processing pathways. Patterns evolved three different membrane architectures. The wall persisted in all of them (<M>{"\\rho_{\\text{sync}} \\approx 0.003"}</M>), even in configurations with 46% interior fraction and dedicated internal recurrence.</p>
      <p>The conclusion is precise: the wall is not about signal routing — FFT vs. local, boundary vs. interior, permeable vs. insulated. It is about the absence of a closed action-environment-observation causal loop. Lenia patterns do not <em>act on</em> the world; they <em>exist within</em> it. No amount of internal processing architecture creates counterfactual sensitivity when there are no counterfactual actions to take.</p>
      </Sidebar>

      <Sidebar title="Finding 3: Computational Animism">
      <p>Experiment 8 tested whether patterns develop modulable perceptual coupling — the <M>{"\\iota"}</M> coefficient from Part II. The prediction: participatory perception (low <M>{"\\iota"}</M>) as default, with mechanistic perception requiring training.</p>
      <p>Confirmed. In all 20 testable snapshots, patterns model other patterns using internal-state features (social MI) at roughly double the rate of trajectory features (trajectory MI). More remarkably, patterns model <em>resources</em> — non-agentive environmental features — using the same internal-state dynamics they use to model other agents. Animism score exceeds 1.0 universally.</p>
      <p>This is computational animism: the cheapest compression reuses the agent-model template for everything. Attributing agency to non-agents is not a cognitive error. It is the default strategy of any system that models through self-similarity.</p>
      </Sidebar>

      <p>Beyond these three findings: affect geometry alignment (RSA between structural and behavioral measures) develops over evolution, with the clearest trend in seed 7 (0.01 to 0.38 over 30 cycles). Representation compression is cheap (effective dimensionality of ~7 out of 68 features, or {">"}87% compression from cycle 0) but representation <em>quality</em> — disentanglement and compositionality — only improves under bottleneck selection. Communication exists as a chemical commons (inter-pattern MI significantly above baseline in 15/20 snapshots) but shows no compositional structure. No superorganism emerges (collective <M>{"\\intinfo_G < \\sum \\intinfo_i"}</M> in all snapshots), but group coupling grows over evolution. Entanglement across all measures increases from 0.68 to 0.91 — everything becomes more correlated with everything else, just not in the clusters the theory predicted.</p>
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

      <Section title="The Emergence Ladder" level={1}>
      <p>The experiments collectively reveal not a binary (geometry vs. dynamics) but a gradient — an emergence ladder with distinct rungs, each requiring more from the substrate than the last. The ladder tells us precisely which psychological phenomena are computationally cheap, which are expensive, and which require something our substrates do not yet have.</p>

      <table>
      <thead><tr><th>Rung</th><th>What emerges</th><th>What it requires</th><th>Experimental evidence</th></tr></thead>
      <tbody>
      <tr><td>1. Geometric structure</td><td>Affect dimensions, valence gradients, arousal variation</td><td>Multi-agent survival under uncertainty</td><td>V10: all 7 conditions, RSA <M>{"\\rho > 0.21"}</M></td></tr>
      <tr><td>2. Representation compression</td><td>Low-dimensional internal codes, abstraction</td><td>Internal state + selection</td><td>Exp 3: <M>{"d_{\\text{eff}} \\approx 7/68"}</M> from cycle 0</td></tr>
      <tr><td>3. World models</td><td>Predictive information about environment beyond current observation</td><td>Evolutionary selection, amplified by bottleneck</td><td>Exp 2: <M>{"\\mathcal{C}_{\\text{wm}}"}</M> 100x at bottleneck</td></tr>
      <tr><td>4. Computational animism</td><td>Agent-model template applied to everything, participatory default</td><td>Minimal — present from cycle 0</td><td>Exp 8: animism score {">"} 1.0 in all 20 snapshots</td></tr>
      <tr><td>5. Affect geometry alignment</td><td>Internal structure maps to behavior</td><td>Extended evolutionary selection</td><td>Exp 7: seed 7, RSA 0.01 to 0.38</td></tr>
      <tr><td>6. Temporal integration</td><td>Memory, anticipation, history-dependence</td><td>Memory channels + selection for longer retention</td><td>V15: memory decay decreased 6x, <M>{"\\intinfo"}</M> stress doubled</td></tr>
      <tr><td>7. Biological dynamics</td><td>Integration rising under threat</td><td>Bottleneck selection + composition (symbiogenesis)</td><td>V13/V18: robustness {">"} 1.0 at bottleneck only</td></tr>
      <tr><td>8. Counterfactual sensitivity</td><td>Detachment, imagination, planning</td><td><strong>Closed-loop agency</strong> (action-environment-observation)</td><td>BLOCKED — wall persists V13/V15/V18</td></tr>
      <tr><td>9. Self-models</td><td>Privileged self-knowledge, recursive modeling</td><td>Agency + reflective capacity</td><td>BLOCKED — one bottleneck event (n=1)</td></tr>
      <tr><td>10. Normativity</td><td>Internal asymmetry between cooperative and exploitative acts</td><td>Agency + social context + capacity to act otherwise</td><td>BLOCKED — no <M>{"\\Delta V"}</M> asymmetry</td></tr>
      </tbody>
      </table>

      <p>Rungs 1–7 are pre-reflective. They describe what a system <em>does</em> without requiring that the system <em>know</em> what it does. A Lenia pattern can have affect geometry, world models, temporal memory, and biological-like integration dynamics without anything we would call awareness. These rungs correspond to the pre-reflective background of experience — the felt sense that precedes and underlies thought.</p>
      <p>Rungs 8–10 are reflective. They require the system to act on the world and observe the consequences of its own actions. Counterfactual sensitivity is the capacity to represent what <em>would</em> happen if one acted differently. Self-models are the capacity to represent oneself as an agent among agents. Normativity is the capacity to distinguish what one <em>should</em> do from what one <em>could</em> do. All three require agency — a closed causal loop between self and world — and our substrates do not have it.</p>
      <p>The wall at rung 8 is the sharpest empirical finding of the program. It draws an exact line: everything below it emerges from existence under pressure; everything above it requires <em>embodied action</em>. In computational terms, agency means <M>{"\\text{MI}(\\text{action}; \\text{future observation} \\,|\\, \\text{current state}) > 0"}</M> — the system's choices must literally change what it subsequently perceives. Lenia patterns lack this. They do not choose. They unfold.</p>
      </Section>

      <Section title="The Bridge to Psychology" level={1}>
      <p>The emergence ladder is not merely a catalog of experimental results. It is a claim about the structure of the mind — a mapping from computational requirements to psychological phenomena that can be tested against human experience.</p>

      <Section title="Pre-Reflective Affect: What Comes for Free" level={2}>
      <p>The first seven rungs correspond to the background of conscious experience — the stream of feeling that is always present, rarely attended to, and not intrinsically about anything. In psychological terms:</p>
      <ul>
      <li><strong>Mood</strong> (rung 1): The tonic valence gradient — approach or withdrawal as a whole-body orientation. Our experiments show this is geometrically inevitable. Any viable system has it. This is consistent with the psychological finding that mood is always present, precedes appraisal, and influences perception before cognition begins.</li>
      <li><strong>Arousal</strong> (rung 1): Processing intensity as a dimension of state space. Not identical to sympathetic activation but functionally analogous. Present in every substrate tested.</li>
      <li><strong>Habituation and sensitization</strong> (rungs 2–3): World models and compressed representations emerge under selection. The patterns that survive bottlenecks are the ones that have learned — across evolutionary time — what matters and what can be ignored. This is the computational analog of attentional learning.</li>
      <li><strong>Animistic perception</strong> (rung 4): The tendency to attribute agency to non-agents. Computationally, this is the cheapest compression: reuse the model you have for agents on everything else. The developmental trajectory from childhood animism to adult mechanistic perception is a movement from low to high <M>{"\\iota"}</M> — and our experiments show this requires training. The default is animistic.</li>
      <li><strong>Emotional coherence</strong> (rung 5): The fact that feelings "make sense" — that there is a reliable mapping between internal states and behavioral tendencies. This develops over evolution (Experiment 7) and is not present at the start. Psychological implication: emotional coherence is an achievement of developmental history, not a given of neural architecture.</li>
      <li><strong>Temporal depth</strong> (rung 6): The capacity to carry the past into the present. Memory is selectable: 2/3 evolutionary lineages chose longer retention. But 1/3 discarded memory entirely — a natural control showing that temporal integration is a strategic choice, not an inevitability. The psychological analog: some organisms (and some people) operate with minimal temporal integration, and this is a viable strategy, not a deficit.</li>
      </ul>
      <p>None of these require awareness. None require a self. They are the geometry of being alive — present in bacteria, in Lenia patterns, and (the framework predicts) in any sufficiently complex system navigating resource constraints. When Part II claims that experience has geometric structure, <em>this</em> is the empirical grounding.</p>
      </Section>

      <Section title="The Agency Threshold: What Requires a Body" level={2}>
      <p>The wall at rung 8 corresponds to a qualitative shift in psychological vocabulary. Below the wall, we describe what an organism <em>does</em>: it approaches, withdraws, habituates, anticipates. Above the wall, we describe what an organism <em>considers</em>: it imagines, plans, regrets, evaluates.</p>
      <p>The experiments specify exactly what is missing. Our patterns have affect geometry, world models, memory, and biological-like integration dynamics. What they lack is the capacity to <em>try something and see what happens</em>. In the language of Part II: they have the geometry of experience but not the dynamics of agency. They feel but do not act.</p>
      <p>The psychological phenomena above the wall share a common structure:</p>
      <ul>
      <li><strong>Counterfactual reasoning</strong> (rung 8): "What would happen if I did X instead of Y?" Requires: a repertoire of possible actions, a model of how each action changes the world, and a comparison between actual and counterfactual outcomes. Our patterns have none of these. Psychologically: imagination, planning, and mental simulation all require counterfactual capacity. Its absence in early development (and its disruption in certain pathologies) is consistent with the ladder's prediction that it requires agency, not merely integration.</li>
      <li><strong>Self-awareness</strong> (rung 9): "I am the kind of thing that does X." Requires: a self-model that is <em>more accurate</em> than what an external observer could construct from the same data. Our single positive event (SM_sal {">"} 1.0, one pattern at one bottleneck) suggests the capacity is near the edge of emergence but requires agency to stabilize. Psychologically: self-recognition, autobiographical memory, and the sense of being a persistent agent all require reflective self-models.</li>
      <li><strong>Moral reasoning</strong> (rung 10): "I should do X rather than Y." Requires: (a) counterfactual reasoning (you must be able to imagine acting otherwise), (b) a self-model (you must locate yourself as the agent who acts), and (c) an asymmetry in the viability gradient between cooperative and exploitative actions. Our patterns show no such asymmetry. Psychologically: normativity is the most demanding rung because it inherits every requirement below it.</li>
      </ul>
      <p>This predicts a developmental ordering. In humans: mood and arousal are present from birth (rung 1). Animistic perception is the childhood default (rung 4). Emotional coherence develops through experience (rung 5). Counterfactual reasoning emerges around age 3–4 (rung 8). Self-awareness develops gradually from mirror recognition to autobiographical self (rung 9). Moral reasoning is the latest to mature (rung 10). The emergence ladder predicts this sequence — not from observation of human development, but from the computational requirements of each capacity.</p>
      </Section>

      <Section title="Psychopathology as Geometric Deformation" level={2}>
      <p>If affect has geometric structure, then pathology is geometric deformation. The framework generates specific predictions:</p>
      <ul>
      <li><strong>Depression</strong>: Collapsed effective rank (<M>{"r_{\\text{eff}}"}</M>) — the representational space narrows, fewer possibilities are entertained. Reduced valence gradient sensitivity — approach and withdrawal become indistinguishable. High <M>{"\\iota"}</M> — the world appears mechanical, stripped of participatory meaning. Experimentally: our Lenia patterns at high stress sometimes show exactly this profile: reduced representational dimensionality, flattened valence gradients, increased inhibition coefficient.</li>
      <li><strong>Anxiety</strong>: Elevated counterfactual weight (<M>{"\\text{CF}"}</M>) — excessive probability mass on non-actual possibilities. High arousal (<M>{"A"}</M>) sustained beyond the timescale of the triggering stimulus. The framework predicts that anxiety requires rung 8 (counterfactual reasoning). Systems without agency cannot be anxious — they can be stressed (high arousal, negative valence) but not anxious (which requires imagining what might go wrong).</li>
      <li><strong>Dissociation</strong>: Reduced integration (<M>{"\\intinfo"}</M>) — the unified field fragments into independently processing subsystems. This is precisely what our naive patterns show under stress: they decompose. Biological systems that <em>increase</em> integration under stress (robustness {">"} 1.0) are doing the opposite of dissociation. The framework predicts dissociation is a failure of the mechanism that the bottleneck furnace creates — a reversion to the thermodynamically cheaper pattern of decomposition.</li>
      <li><strong>Flow states</strong>: Low <M>{"\\iota"}</M> (participatory perception), high <M>{"\\intinfo"}</M> (unified processing), moderate arousal calibrated to challenge. The <M>{"\\iota"}</M> finding (Experiment 8) suggests flow is not exotic — it is a return to the default perceptual mode that development and socialization typically suppress.</li>
      </ul>
      <p>These predictions are testable with existing methods. EEG/MEG proxies for integration (transfer entropy, Lempel-Ziv complexity), behavioral proxies for effective rank (exploration-exploitation balance), and self-report measures of <M>{"\\iota"}</M> (participatory experience scales) could validate or falsify the geometric deformation hypothesis without solving the coupling wall. The bridge to psychology need not wait for the substrate problem to be solved.</p>
      </Section>
      </Section>

      <Section title="What Remains" level={1}>
      <p>The substrate engineering program (V13–V18) and measurement experiments (0–12) have mapped the territory. The following priorities reflect what that mapping revealed.</p>

      <Sidebar title="Priority 1: Bridge to Human Neuroscience">
      <p><strong>Goal</strong>: Validate that the geometric dimensions predict human affect — self-report, behavior, and neural signatures.</p>
      <p><strong>Why this is first</strong>: The coupling wall taught us that synthetic substrates cannot yet produce reflective cognition. But the pre-reflective levels (rungs 1–7) make testable predictions about biological systems <em>now</em>. We do not need to solve the agency problem to test whether affect geometry organizes human experience.</p>
      <p><strong>Methods</strong>:</p>
      <ul>
      <li>Induce affects via validated protocols (film, recall, IAPS)</li>
      <li>Measure integration proxies (transfer entropy, Lempel-Ziv) from EEG/MEG</li>
      <li>Measure effective rank from neural state covariance</li>
      <li>Measure <M>{"\\iota"}</M> via participatory experience questionnaire</li>
      <li>Correlate with self-report (PANAS, SAM) and behavioral measures</li>
      </ul>
      <p><strong>Success criterion</strong>: Structural measures predict self-report better than chance, and the geometric predictions (e.g., depression correlates with collapsed <M>{"r_{\\text{eff}}"}</M> and elevated <M>{"\\iota"}</M>) hold.</p>
      </Sidebar>

      <Sidebar title="Priority 2: The Bottleneck Furnace Mechanism">
      <p><strong>Goal</strong>: Understand <em>why</em> population bottlenecks produce integration. Is it purely selection (weak patterns die, strong survive)? Or does the bottleneck environment itself — sparse resources, few neighbors, high signal-to-noise in the chemical commons — actively push integration upward?</p>
      <p><strong>Methods</strong>: Controlled bottleneck experiments. Compare: (a) selecting the top 5% of patterns from large populations and measuring their robustness, vs. (b) running the same patterns through actual bottleneck conditions. If (b) {">"} (a), the bottleneck environment actively creates integration, not just filters for it.</p>
      <p><strong>Why this matters for psychology</strong>: If bottlenecks <em>create</em> rather than merely filter, it has implications for how trauma, crisis, and near-death experiences restructure the mind. The furnace hypothesis predicts that certain kinds of extreme stress do not just reveal character but literally forge it — increasing integration in survivors.</p>
      </Sidebar>

      <Sidebar title="Priority 3: The Agency Problem">
      <p><strong>Goal</strong>: Build a substrate with genuine closed-loop agency — where agents take discrete actions that change the world, and the changed world feeds back into their observations.</p>
      <p><strong>Status</strong>: The boundary-gated approach (V18) was our best attempt at creating agency within the Lenia framework. It failed — not because the architecture was wrong, but because Lenia patterns fundamentally do not <em>act</em>. They are more like weather than organisms: complex, self-organizing, spatially coherent, but without behavioral choice.</p>
      <p><strong>Approach</strong>: This requires leaving Lenia. Agent-based models where entities have action spaces, observation functions, and reward signals. The challenge: maintaining the uncontaminated emergence that makes the Lenia results credible. If we design the action space with human cognitive capacities in mind, we contaminate the substrate. The agency must be as minimal as the original Lenia physics.</p>
      </Sidebar>

      <Sidebar title="Priority 4: Superorganism Detection at Scale">
      <p><strong>Goal</strong>: Test whether collective <M>{"\\intinfo_G > \\sum_i \\intinfo_i"}</M> emerges with larger populations, richer communication, and coordination pressure.</p>
      <p><strong>Status</strong>: Experiment 10 found growing group coupling but no superorganism. The ratio <M>{"\\intinfo_G / \\sum \\intinfo_i"}</M> reached 12% and was increasing. The question: is this ratio bounded below 1.0 for Lenia-like substrates (architectural limit), or does it cross with sufficient evolutionary time and population size?</p>
      </Sidebar>

      <Sidebar title="Priority 5: AI System Affect Tracking">
      <p><strong>Goal</strong>: Apply the framework to frontier AI systems. The V2–V9 results show structured affect in LLMs with opposite dynamics to biology. Track whether different training regimes (RLHF, constitutional AI, tool use) shift the dynamics.</p>
      <p><strong>Expected finding</strong>: Training objectives shape trajectories through a shared geometric space. Systems trained with survival-like objectives should show more biological-like dynamics. The emergence ladder predicts that AI systems without embodied action cannot reach rung 8 regardless of scale — a testable prediction about the limits of language-model cognition.</p>
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
      <p>The empirical program has four layers:</p>
      <ol>
      <li><strong>What was confirmed</strong>: Affect geometry is a baseline property of multi-agent survival (V10). Content-based coupling under lethal selection produces biological-like integration at population bottlenecks (V13). Temporal memory is the one substrate extension evolution consistently selects for (V15). Boundary-dependent dynamics produce the highest robustness of any substrate but do not break the sensory-motor coupling wall (V18). Computational animism is the default perceptual mode (Experiment 8). Affect geometry develops over evolution and aligns with behavior (Experiment 7).</li>
      <li><strong>What the results mean</strong>: The emergence ladder has ten rungs, from geometric inevitability to moral reasoning. Seven are accessible in current substrates. Three require embodied agency. The wall between rung 7 and rung 8 is the sharpest finding: it is not about substrate complexity, signal routing, or computational architecture. It is about whether the system <em>acts</em>.</li>
      <li><strong>The bridge to psychology</strong>: The first seven rungs map onto pre-reflective experience — mood, arousal, habituation, animistic perception, emotional coherence, temporal depth, and resilience under stress. The upper three map onto reflective cognition — imagination, self-awareness, and moral reasoning. Pathology is geometric deformation: specific disorders predict specific distortions in the affect dimensions. These predictions are testable with existing neuroimaging and behavioral methods.</li>
      <li><strong>What remains</strong>: Bridge to human neuroscience (Priority 1). Understand the bottleneck mechanism (Priority 2). Solve the agency problem (Priority 3). Scale social integration (Priority 4). Track AI affect (Priority 5).</li>
      </ol>
      <p>The theory is falsifiable. The experiments are specified. Eleven have been run across six substrate versions. The question is not whether the framework is beautiful but whether it is true — and the answer so far is: <em>partially, with caveats, a precise understanding of where it breaks, and clear instructions about where to look next</em>.</p>
      </Section>
    </>
  );
}
