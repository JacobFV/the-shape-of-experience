// WORK IN PROGRESS: This is active research, not a finished publication.
// Content is incomplete, speculative, and subject to change.

import { Eq, Experiment, Logos, M, OpenQuestion, Section, Sidebar } from '@/components/content';

export const metadata = {
  slug: 'appendix-experiments',
  title: 'Appendix: Experiment Catalog',
  shortTitle: 'Experiments',
};

export default function AppendixExperiments() {
  return (
    <>
      <Logos>
      <p>Every claim in this book is either tested, testable, or honestly labeled as speculative. This appendix catalogs the full experimental program: what has been run, what the results show, and what the results mean. Eleven measurement experiments on an uncontaminated substrate. Five substrate iterations beyond the baseline. Three seeds each. The data is in.</p>
      </Logos>

      <Section title="The Substrate Ladder" level={1}>
      <p>Before the measurement experiments, we had to build a substrate worth measuring. Seven versions, each adding one capability, tracking whether evolution selects for it.</p>

      <Section title="V10: MARL Forcing Function Ablation" level={2}>
      <p><strong>Question</strong>: Do forcing functions create geometric affect alignment?</p>
      <p><strong>Result</strong>: No. Seven conditions (full model plus six single-ablation), three seeds each, 200,000 steps on GPU. All conditions show significant alignment (RSA <M>{"\\rho > 0.21"}</M>, <M>{"p < 0.0001"}</M>). Removing forcing functions slightly <em>increases</em> alignment.</p>
      <p><strong>Implication</strong>: Geometry is cheap. The forcing functions hypothesis was downgraded from theorem to hypothesis.</p>
      </Section>

      <Section title="V11.0-V11.7: Lenia CA Evolution Series" level={2}>
      <p><strong>Question</strong>: Can evolution under survival pressure produce biological-like integration dynamics?</p>
      <table>
      <thead><tr><th>Version</th><th><M>{"\\Delta\\intinfo"}</M> (drought)</th><th>Key lesson</th></tr></thead>
      <tbody>
      <tr><td>V11.0 (naive)</td><td>-6.2%</td><td>Decomposition baseline</td></tr>
      <tr><td>V11.1 (homogeneous evolution)</td><td>-6.0%</td><td>Selection alone insufficient</td></tr>
      <tr><td>V11.2 (heterogeneous chemistry)</td><td>-3.8%</td><td>+2.1pp shift from diverse viability manifolds</td></tr>
      <tr><td>V11.7 (curriculum training)</td><td>+1.2 to +2.7pp generalization</td><td>Only intervention improving novel-stress response</td></tr>
      </tbody>
      </table>
      <p><strong>Implication</strong>: Training regime matters more than substrate complexity. The locality ceiling: convolutional physics cannot produce active self-maintenance under severe threat.</p>
      </Section>

      <Section title="V12: Attention-Based Lenia" level={2}>
      <p><strong>Question</strong>: Does state-dependent interaction topology enable the biological integration pattern?</p>
      <p><strong>Result</strong>: Partially. Evolvable attention: <M>{"\\intinfo"}</M> increase in 42% of cycles (vs 3% for convolution). +2.0pp shift — largest single-intervention effect. But robustness stabilizes near 1.0 without further improvement.</p>
      <p><strong>Implication</strong>: Attention is necessary but not sufficient. The system reaches the integration threshold without crossing it.</p>
      </Section>

      <Section title="V13: Content-Based Coupling" level={2}>
      <p><strong>Substrate</strong>: FFT convolution + content-similarity modulation. Cells couple more strongly with cells sharing state-features — chemical affinity rather than cognitive attention.</p>
      <Eq>{"K_i(j) = K_{\\text{base}}(|i-j|) \\cdot \\sigma\\!\\bigl(\\langle h(s_i),\\, h(s_j) \\rangle - \\tau\\bigr)"}</Eq>
      <p>Three seeds, 30 cycles each (<M>{"C{=}16"}</M>, <M>{"N{=}128"}</M>). Mean robustness 0.923, peak 1.052 at population bottleneck. Robustness {">"} 1.0 only when population drops below ~50. Two evolutionary strategies: open coupling (<M>{"\\tau \\to 0"}</M>, large stable populations) vs selective coupling (<M>{"\\tau \\to 0.86"}</M>, volatile with occasional robustness {">"} 1.0).</p>
      <p>This became the foundation substrate for all measurement experiments (Experiments 0–12).</p>
      </Section>

      <Section title="V14: Chemotaxis" level={2}>
      <p><strong>Addition</strong>: Motor channels enabling directed foraging. Velocity field from resource gradients gated by the last two of <M>{"C{=}16"}</M> channels.</p>
      <p><strong>Result</strong>: Patterns move 3.5–5.6 pixels/cycle toward resources. Motor sensitivity evolves — not just maximized but modulated by selection. Robustness comparable to V13 (~0.90–0.95).</p>
      </Section>

      <Section title="V15: Temporal Memory" level={2}>
      <p><strong>Addition</strong>: Two exponential-moving-average memory channels storing slow statistics of the pattern's history. Oscillating resource patches reward anticipation.</p>
      <p><strong>Result</strong>: Evolution selected for longer memory in 2/3 seeds — memory decay constants decreased 6x in seed 42, meaning the system retained information over longer timescales. Under bottleneck pressure, <M>{"\\intinfo"}</M> stress response doubled (0.231 to 0.434). Peak robustness 1.070.</p>
      <p><strong>Key finding</strong>: Temporal integration is fitness-relevant. This was the only substrate addition that evolution consistently selected for.</p>
      </Section>

      <Section title="V16: Hebbian Plasticity (Negative Result)" level={2}>
      <p><strong>Addition</strong>: Local Hebbian learning rules allowing each spatial location to modify its coupling structure in response to experience.</p>
      <p><strong>Result</strong>: Mean robustness dropped to 0.892 — lowest of V13+. Zero cycles exceeded 1.0. Coupling spatial variance collapsed. Plasticity added noise faster than selection could filter it.</p>
      <p><strong>Lesson</strong>: Simple learning rules are too blunt. The extra degrees of freedom overwhelm the selection signal.</p>
      </Section>

      <Section title="V17: Quorum Signaling (Mixed Result)" level={2}>
      <p><strong>Addition</strong>: Two diffusible signal fields mediating inter-pattern coordination, analogous to bacterial quorum sensing.</p>
      <p><strong>Result</strong>: Produced the highest-ever single-cycle robustness (1.125) in a seed that evolved hyper-sensitive coupling modulation at a population of 2. But 2/3 seeds evolved to <em>suppress</em> signaling entirely — emission collapsed to near-zero or sensitivity increased until the system was effectively deaf.</p>
      <p><strong>Lesson</strong>: Signaling is costly in large populations, beneficial only at extreme bottlenecks. The substrate finds it easier to suppress a coordination mechanism than to use it constructively.</p>
      </Section>

      <Section title="V18: Boundary-Dependent Lenia" level={2}>
      <p><strong>Addition</strong>: Insulation field computed via iterated erosion and sigmoid creates a genuine boundary/interior distinction. Dual-signal update: external FFT signals gated by <M>{"(1 - \\text{insulation})"}</M>, internal short-range recurrence gated by <M>{"\\text{insulation}"}</M>. Small patterns have insulation ≈ 0 (pure V15 behavior). Large patterns develop insulated cores with autonomous internal dynamics. New evolvable parameters: <code>boundary_width</code>, <code>insulation_beta</code>, <code>internal_gain</code>, <code>activity_threshold</code>.</p>
      <p><strong>Hypothesis</strong>: Boundary-gated signals would create the reactive phase needed for counterfactual detachment — patterns would update from boundary observations rather than full-grid FFT, making the reactive-to-autonomous transition measurable and selectable.</p>
      <p><strong>Result</strong>: Three seeds, 30 cycles, <M>{"C{=}16"}</M>, <M>{"N{=}128"}</M>. Mean robustness 0.969 — highest of any substrate. Peak 1.651 (seed 42). 33% of cycles show <M>{"\\intinfo"}</M> increase under stress. Three evolutionary strategies emerged:</p>
      <ul>
      <li><strong>Seed 42</strong>: Thin permeable membrane, moderate interior fraction (46%), <code>internal_gain</code> halved</li>
      <li><strong>Seed 123</strong>: Thick membrane, large interior (32%), high gain — closest to the autonomous archetype</li>
      <li><strong>Seed 7</strong>: Filaments with no interior (0.1%) — maximally reactive, no internal processing</li>
      </ul>
      <p><strong>Surprise</strong>: <code>internal_gain</code> evolved <em>down</em> in all three seeds (1.0 → ~0.6). The prediction was that patterns would develop stronger internal processing as they insulated themselves from external noise. Instead, evolution preferred permeable membranes. <code>boundary_width</code> converged to the minimum allowed value (0.5). External sensing was more valuable than internal rumination at this grid scale.</p>
      <p><strong>Implication</strong>: The architectural modification achieved its engineering goal (highest robustness) but not its theoretical goal (breaking the coupling wall). See the Sensory-Motor Coupling Wall section.</p>
      </Section>

      <Section title="Cross-Version Summary" level={2}>
      <table>
      <thead><tr><th>Version</th><th>Mean Robustness</th><th>Max Robustness</th><th>{">"} 1.0 Cycles</th><th>Verdict</th></tr></thead>
      <tbody>
      <tr><td>V13 (content coupling)</td><td>0.923</td><td>1.052</td><td>3/90</td><td>Foundation substrate</td></tr>
      <tr><td>V14 (+ chemotaxis)</td><td>~0.91</td><td>~0.95</td><td>~1/90</td><td>Motion evolves</td></tr>
      <tr><td>V15 (+ memory)</td><td>0.907</td><td>1.070</td><td>3/90</td><td>Best dynamics</td></tr>
      <tr><td>V16 (+ plasticity)</td><td>0.892</td><td>0.974</td><td>0/90</td><td>Negative</td></tr>
      <tr><td>V17 (+ signaling)</td><td>0.892</td><td>1.125</td><td>1/90</td><td>Suppressed</td></tr>
      <tr><td>V18 (boundary-dependent)</td><td>0.969</td><td>1.651</td><td>~10/90</td><td>Best robustness; gain↓</td></tr>
      </tbody>
      </table>
      <p>V18 achieved the highest mean robustness of any substrate — but evolution did not use the insulation mechanism as intended. Internal gain evolved down across all seeds; boundary width converged to minimum. The architectural improvement did not break the sensory-motor coupling wall (see below). V15 remains the best substrate for dynamics; V18 for raw robustness.</p>
      </Section>
      </Section>

      <Section title="The Emergence Experiment Program" level={1}>
      <p>Eleven measurement experiments on V13 snapshots, testing whether the capacities the preceding six parts describe — world modeling, abstraction, communication, counterfactual reasoning, self-modeling, affect structure, perceptual mode, normativity, social integration — emerge in a substrate with zero exposure to human affect concepts. All experiments: 3 seeds, 7 snapshots per seed (every 5 evolutionary cycles), 50 recording steps per snapshot.</p>

      <Section title="Experiment 0: Substrate Engineering" level={2}>
      <p><strong>Status</strong>: Complete. V13 content-based coupling Lenia with lethal resource dynamics. The substrate sustains 50–180 patterns across 30 evolution cycles with content-similarity modulation, curriculum stress schedule, and population rescue. <M>{"\\tau"}</M> and gate steepness <M>{"\\beta"}</M> are evolvable. Foundation for all subsequent experiments.</p>
      </Section>

      <Section title="Experiment 1: Emergent Existence" level={2}>
      <p><strong>Status</strong>: Complete. Patterns persist, maintain boundaries, respond to perturbation. Established by V11–V12, confirmed in V13. The operational definition: a pattern exists when its internal correlations exceed background and it maintains identity over time.</p>
      </Section>

      <Section title="Experiment 2: Emergent World Model" level={2}>
      <p><strong>Question</strong>: When does a pattern's internal state carry predictive information about the environment beyond what's available from current observations?</p>
      <p><strong>Method</strong>: Prediction gap <M>{"\\mathcal{W}(\\tau) = \\text{MSE}[f_{\\text{env}}] - \\text{MSE}[f_{\\text{full}}]"}</M> using Ridge regression with 5-fold CV. Features: 68-dim internal state, 36-dim boundary ring, 18-dim environment target. <M>{"\\tau \\in \\{1, 2, 5, 10, 20\\}"}</M> recording steps.</p>
      <table>
      <thead><tr><th>Seed</th><th><M>{"\\mathcal{C}_{\\text{wm}}"}</M> (early)</th><th><M>{"\\mathcal{C}_{\\text{wm}}"}</M> (late)</th><th><M>{"H_{\\text{wm}}"}</M> (late)</th><th>% with WM</th></tr></thead>
      <tbody>
      <tr><td>123</td><td>0.0004</td><td><strong>0.0282</strong></td><td>20.0</td><td>100%</td></tr>
      <tr><td>42</td><td>0.0002</td><td>0.0002</td><td>5.3</td><td>40%</td></tr>
      <tr><td>7</td><td>0.0010</td><td>0.0002</td><td>7.9</td><td>60%</td></tr>
      </tbody>
      </table>
      <p><strong>Finding</strong>: World model signal is present but weak in the general population (<M>{"\\mathcal{C}_{\\text{wm}} \\sim 10^{-4}"}</M>). Seed 123 at population bottleneck (cycle 29, 1 survivor): <M>{"\\mathcal{C}_{\\text{wm}} = 0.028"}</M>, roughly 100x the population average. World models emerge but are amplified by bottleneck selection, not gradual evolution.</p>
      <p><strong>V15 re-run</strong>: Memory channels improved world model capacity ~12x on seed 42 (0.00244 vs 0.0002). V13 bottleneck peak (0.028) remains the dominant signal.</p>
      <p><strong>V18 re-run</strong>: <M>{"\\mathcal{C}_{\\text{wm}}"}</M> range 0.0001–0.007. Peak 0.0067 (seed 123, cycle 15). Slight improvement on some snapshots but inconsistent across seeds and cycles. The V13 bottleneck peak (0.028) remains the dominant signal.</p>
      </Section>

      <Section title="Experiment 3: Internal Representation Structure" level={2}>
      <p><strong>Question</strong>: When do patterns develop low-dimensional, compositional representations?</p>
      <p><strong>Method</strong>: PCA on standardized 68-dim internal state. Effective dimensionality <M>{"d_{\\text{eff}} = (\\text{tr}\\,\\Sigma)^2 / \\text{tr}(\\Sigma^2)"}</M>. Disentanglement <M>{"\\mathcal{D}"}</M> = mean max-<M>{"R^2"}</M> between PCA dims and environmental features. Compositionality <M>{"K_{\\text{comp}}"}</M> = linear composition prediction error.</p>
      <table>
      <thead><tr><th>Seed</th><th><M>{"d_{\\text{eff}}"}</M> (early to late)</th><th><M>{"\\mathcal{A}"}</M> (early to late)</th><th><M>{"\\mathcal{D}"}</M> (early to late)</th><th><M>{"K_{\\text{comp}}"}</M> (early to late)</th></tr></thead>
      <tbody>
      <tr><td>123</td><td>6.6 to <strong>5.6</strong></td><td>0.90 to <strong>0.92</strong></td><td>0.27 to <strong>0.38</strong></td><td>0.20 to <strong>0.12</strong></td></tr>
      <tr><td>42</td><td>7.3 to 7.5</td><td>0.89 to 0.89</td><td>0.23 to 0.23</td><td>0.23 to 0.25</td></tr>
      <tr><td>7</td><td>7.7 to 8.8</td><td>0.89 to 0.87</td><td>0.24 to 0.22</td><td>0.20 to 0.27</td></tr>
      </tbody>
      </table>
      <p><strong>Finding</strong>: Compression is cheap — <M>{"d_{\\text{eff}} \\approx 7"}</M> out of 68 dimensions ({">"} 87% compression) from cycle 0. But representation <em>quality</em> — disentanglement and compositionality — only improves under bottleneck selection (seed 123). In the general population, no trend. Geometry is cheap for representations too.</p>
      </Section>

      <Section title="Experiment 4: Emergent Language" level={2}>
      <p><strong>Question</strong>: When do patterns develop structured, compositional communication?</p>
      <p><strong>Method</strong>: Inter-pattern mutual information (MI) via chemical commons. Compositionality via topographic similarity <M>{"\\rho_{\\text{topo}}"}</M>.</p>
      <table>
      <thead><tr><th>Seed</th><th>MI significant</th><th>MI range</th><th><M>{"\\rho_{\\text{topo}}"}</M> significant</th></tr></thead>
      <tbody>
      <tr><td>123</td><td>4/6</td><td>0.019–0.039</td><td>0/6</td></tr>
      <tr><td>42</td><td>7/7</td><td>0.024–0.030</td><td>0/7</td></tr>
      <tr><td>7</td><td>4/7</td><td>0.023–0.055</td><td>0/7</td></tr>
      </tbody>
      </table>
      <p><strong>Finding</strong>: Chemical commons, not proto-language. Patterns share information through the chemical medium — MI above shuffled baseline in 15/20 snapshots. But <M>{"\\rho_{\\text{topo}} \\approx 0"}</M> everywhere: communication is unstructured broadcast, not language-like signaling. MI increases over evolution (seed 7: 0.025 to 0.037), but compositional structure never appears.</p>
      </Section>

      <Section title="Experiment 5: Counterfactual Detachment" level={2}>
      <p><strong>Question</strong>: When do patterns decouple from external driving and run offline world model rollouts?</p>
      <p><strong>Method</strong>: External synchrony <M>{"\\rho_{\\text{sync}}"}</M> between internal updates and boundary input. Imagination capacity <M>{"\\mathcal{I}_{\\text{img}}"}</M> = predictive advantage of detached-mode trajectories.</p>
      <p><strong>Result: Null.</strong> <M>{"\\rho_{\\text{sync}} \\approx 0"}</M> from cycle 0 in all seeds. Patterns are inherently internally driven — detachment is the default, not an achievement. The FFT convolution kernel integrates over the full <M>{"128 \\times 128"}</M> grid, so boundary observations are a negligible fraction of the information driving updates. There is no reactive-to-autonomous transition because the starting point is already autonomous.</p>
      <p><strong>V15 re-run</strong>: Wall persists. Motor channels do not create boundary-dependent dynamics. <M>{"\\rho_{\\text{sync}} \\approx 0"}</M> on V15.</p>
      <p><strong>V18 re-run</strong>: Wall persists despite boundary-gated signal architecture. <M>{"\\rho_{\\text{sync}} \\approx 0.003"}</M> across all seeds. Patterns with up to 46% interior fraction and dedicated internal recurrence channels still show no coupling to boundary observations. The wall is definitively about agency (action→observation loops), not signal routing.</p>
      </Section>

      <Section title="Experiment 6: Self-Model Emergence" level={2}>
      <p><strong>Question</strong>: When does a pattern predict itself better than an external observer can?</p>
      <p><strong>Method</strong>: Self-effect ratio <M>{"\\rho_{\\text{self}}"}</M>, self-prediction score SM, self-model salience <M>{"\\text{SM}_{\\text{sal}}"}</M>.</p>
      <p><strong>Result: Weak signal at bottleneck only.</strong> <M>{"\\rho_{\\text{self}} \\approx 0"}</M> everywhere. Self-prediction is trivially present (any spatially-coherent pattern predicts itself somewhat), but self-model <em>salience</em> — privileged self-knowledge exceeding environment-knowledge — appears exactly once: seed 123, cycle 20 (3 patterns at bottleneck), where one pattern achieves <M>{"\\text{SM}_{\\text{sal}} = 2.28"}</M>.</p>
      <p><strong>V15 re-run</strong>: Wall persists. <M>{"\\rho_{\\text{self}}"}</M> slightly higher (max 0.08 vs 0.05) but same regime.</p>
      <p><strong>V18 re-run</strong>: <M>{"\\rho_{\\text{self}} \\approx 0.03{-}0.11"}</M>, similar to the V15 range. <M>{"\\text{SM}_{\\text{sal}}"}</M> remains trivially inflated (division-by-zero artifact in low-population snapshots). No improvement from boundary-dependent dynamics.</p>
      </Section>

      <Section title="Experiment 7: Affect Geometry Verification" level={2}>
      <p><strong>Question</strong>: Does the geometric affect structure predicted by the thesis actually appear? RSA between structural affect (Space A: valence, arousal, integration, effective rank, counterfactual weight, self-model salience) and behavioral affect (Space C: approach/avoid, activity, growth, stability).</p>
      <table>
      <thead><tr><th>Seed</th><th><M>{"\\rho(A,C)"}</M> range</th><th>Significant (<M>{"p < 0.05"}</M>)</th><th>Trend</th></tr></thead>
      <tbody>
      <tr><td>123</td><td>-0.09 to 0.72</td><td>2/5 testable</td><td>Strong at low pop (0.72)</td></tr>
      <tr><td>42</td><td>-0.17 to 0.39</td><td>4/7 (mixed)</td><td>No clear trend</td></tr>
      <tr><td>7</td><td>0.01 to 0.38</td><td>5/7</td><td><strong>Increasing</strong> (0.01 to 0.24)</td></tr>
      </tbody>
      </table>
      <p><strong>Finding</strong>: Structural affect aligns with behavior in 8/19 testable snapshots. Seed 7 shows the clearest evolutionary trend: alignment increases from near-zero to consistently significant over 30 cycles. Structure determines behavior, but only after selection shapes the mapping. A-B alignment (from Experiment 4's <M>{"\\rho_{\\text{topo}}"}</M>) is null — structure maps to behavior but not to communication.</p>
      <p><strong>V15 re-run</strong>: Weaker than V13 (2/21 significant vs 8/19). Extra channels may add noise to the RSA computation.</p>
      <p><strong>V18 re-run</strong>: Mixed. Seed 7 shows development (RSA 0.22 to 0.30, <M>{"p < 0.01"}</M> at cycle 29). 5/18 testable snapshots significant — better than V15 (2/21) but comparable to V13 (8/19). The higher robustness substrate does not consistently improve affect geometry alignment.</p>
      </Section>

      <Section title="Experiment 8: Perceptual Mode and Computational Animism" level={2}>
      <p><strong>Question</strong>: Do patterns develop modulable perceptual coupling — switching between modeling others as agents (low <M>{"\\iota"}</M>) vs. objects (high <M>{"\\iota"}</M>)?</p>
      <p><strong>Method</strong>: <M>{"\\iota = 1 - (\\text{participatory MI} / \\text{total model MI})"}</M>. Animism score = MI applied to non-agentive objects / MI applied to agents.</p>
      <table>
      <thead><tr><th>Metric</th><th>Seed 123</th><th>Seed 42</th><th>Seed 7</th></tr></thead>
      <tbody>
      <tr><td><M>{"\\iota"}</M> (mean)</td><td>0.27–0.44</td><td>0.27–0.41</td><td>0.31–0.35</td></tr>
      <tr><td><M>{"\\iota"}</M> trajectory</td><td>0.32 to 0.29</td><td>0.41 to 0.27</td><td>0.31 to 0.32</td></tr>
      <tr><td>MI social / MI trajectory</td><td>~2.1x</td><td>~2.3x</td><td>~2.2x</td></tr>
      <tr><td>Animism score</td><td>1.28–2.10</td><td>1.60–2.16</td><td>1.10–2.02</td></tr>
      </tbody>
      </table>
      <p><strong>Finding: Confirmed.</strong> Default is participatory (<M>{"\\iota \\approx 0.30"}</M>). Patterns model others' internal chemistry at roughly double the rate of trajectory features. <M>{"\\iota"}</M> decreases over evolution (seed 42: 0.41 to 0.27) — selection favors participatory perception.</p>
      <p>Animism score {">"} 1.0 in all 20 testable snapshots. Patterns model <em>resources</em> — non-agentive environmental features — using the same internal-state dynamics they use to model other agents. Computational animism is the default: reusing the agent-model template for everything is the cheapest compression.</p>
      </Section>

      <Section title="Experiment 9: Proto-Normativity" level={2}>
      <p><strong>Question</strong>: Does the viability gradient generate structural normativity — an internal state difference when patterns act cooperatively vs. exploitatively?</p>
      <p><strong>Result: Null.</strong> <M>{"\\Delta\\intinfo"}</M>(cooperative - competitive) and <M>{"\\Delta V"}</M> show no consistent direction or significance (2/20 significant total). Patterns don't differentiate cooperative from competitive social contexts internally.</p>
      <p>However: <M>{"\\intinfo_{\\text{social}} \\gg \\intinfo_{\\text{isolated}}"}</M> (~4.9 vs ~3.1). Being near other patterns increases integration regardless of interaction type. This is a social integration effect, not a normative one. Normativity requires agency — the capacity to act otherwise — which V13 patterns lack.</p>
      </Section>

      <Section title="Experiment 10: Social-Scale Integration" level={2}>
      <p><strong>Question</strong>: Does collective <M>{"\\intinfo_G > \\sum_i \\intinfo_i"}</M>? Does the group become a superorganism?</p>
      <table>
      <thead><tr><th>Metric</th><th>Seed 123</th><th>Seed 42</th><th>Seed 7</th></tr></thead>
      <tbody>
      <tr><td>Superorganism (<M>{"\\intinfo_G > \\sum\\intinfo_i"}</M>)</td><td>0/5</td><td>0/7</td><td>0/7</td></tr>
      <tr><td>Super ratio range</td><td>0.009–0.069</td><td>0.029–0.090</td><td>0.052–0.123</td></tr>
      <tr><td><M>{"\\intinfo_G"}</M> trajectory</td><td>0.18 to 5.46</td><td>1.34 to 4.99</td><td>4.74 to 8.55</td></tr>
      </tbody>
      </table>
      <p><strong>Finding</strong>: No superorganism. <M>{"\\intinfo_G < \\sum \\intinfo_i"}</M> in all 19 testable snapshots — individual integration always far exceeds group integration (ratio 1–12%). But the ratio increases over evolution: seed 7 goes from 6.1% to 12.3%. Group coupling roughly doubles. The system is moving toward but not reaching the superorganism threshold.</p>
      </Section>

      <Section title="Experiment 11: Entanglement Analysis" level={2}>
      <p><strong>Question</strong>: Are world models, abstraction, communication, detachment, and self-modeling separable phase transitions or entangled aspects of one process?</p>
      <p><strong>Method</strong>: Correlation matrix across 24 measures for each seed-cycle snapshot. Hierarchical clustering on absolute correlation.</p>
      <p><strong>Finding</strong>: Four clusters emerge — but not the clusters the theory predicted:</p>
      <ol>
      <li><strong>Robustness cluster</strong>: robustness, <M>{"\\intinfo"}</M>-increase, disentanglement</li>
      <li><strong>Large coupling cluster</strong> (14 measures): <M>{"\\mathcal{C}_{\\text{wm}}"}</M>, MI, <M>{"\\iota"}</M>, <M>{"\\intinfo_G"}</M>, animism, coherence — most metrics driven by a single factor (likely population-mediated selection)</li>
      <li><strong>Dimensionality cluster</strong>: <M>{"d_{\\text{eff}}"}</M>, abstraction level</li>
      <li><strong>Self-coupling cluster</strong>: <M>{"\\rho_{\\text{self}}"}</M>, <M>{"\\rho_{\\text{topo}}"}</M></li>
      </ol>
      <p>The predicted co-emergence of <M>{"\\mathcal{C}_{\\text{wm}}"}</M>, abstraction, and imagination (mean <M>{"r = 0.19"}</M>) was not confirmed. But overall entanglement increases: mean absolute correlation grows from 0.68 to 0.91 over evolution. As selection proceeds, everything becomes more correlated with everything else — just not in the specific clusters the theory predicted.</p>
      </Section>

      <Section title="Experiment 12: Identity Thesis Capstone" level={2}>
      <p><strong>Question</strong>: Does the full program — world models, self-models, communication, affect geometry — hold in a system with zero human contamination?</p>
      <table>
      <thead><tr><th>Criterion</th><th>Status</th><th>Strength</th></tr></thead>
      <tbody>
      <tr><td>World models (<M>{"\\mathcal{C}_{\\text{wm}} > 0"}</M>)</td><td>Met</td><td>Weak (strong at bottleneck)</td></tr>
      <tr><td>Self-models (SM {">"} 0)</td><td>Met</td><td>Weak (n=1 event)</td></tr>
      <tr><td>Communication (<M>{"C_{ij} > 0"}</M>)</td><td>Met</td><td>Moderate (15/21 significant)</td></tr>
      <tr><td>Affect dimensions measurable</td><td>Met</td><td><strong>Strong</strong> (84/84 valid)</td></tr>
      <tr><td>Affect geometry (RSA {">"} 0)</td><td>Met</td><td>Moderate (9/19 significant)</td></tr>
      <tr><td>Tripartite alignment</td><td>Met</td><td>Partial (A-C positive, A-B null)</td></tr>
      <tr><td>Perturbation response</td><td>Met</td><td>Moderate (robustness 0.923, {">"} 1.0 at bottleneck)</td></tr>
      </tbody>
      </table>
      <p><strong>Verdict</strong>: All seven criteria met, most at moderate or weak strength. The strongest finding: the geometric framework produces valid, measurable quantities in every snapshot (criterion 4). The weakest: world models and self-models, which are detectable but only meaningfully strong at population bottlenecks. The identity thesis is supported at the geometric level. The dynamical claims remain undertested, blocked by the sensory-motor coupling wall.</p>
      </Section>
      </Section>

      <Section title="The Sensory-Motor Coupling Wall" level={1}>
      <p>Three experiments returned null results (5, 6, 9) and two others showed weakness (2, 7 under V15). All hit the same architectural limitation.</p>
      <p>The FFT convolution kernel integrates over the full <M>{"128 \\times 128"}</M> grid. Boundary observations are a negligible fraction of the information driving updates. Patterns are inherently internally driven (<M>{"\\rho_{\\text{sync}} \\approx 0"}</M> from cycle 0). There is no reactive-to-autonomous transition because the starting point is already autonomous.</p>
      <p>V15's motor channels (chemotaxis) and memory channels did not fix this. V18 introduced boundary-dependent dynamics — an insulation field creating distinct boundary and interior signal domains — specifically to break this wall. Result: wall persists. <M>{"\\rho_{\\text{sync}} \\approx 0.003"}</M> across all seeds and cycles, even in seed 123 with 32% interior fraction and high internal gain. The wall is not about signal routing architecture (FFT vs. local, boundary vs. interior). It is about the absence of a genuine action→environment→observation causal loop. Lenia patterns don't act on the world — they exist within it. No amount of internal processing architecture creates counterfactual sensitivity when there are no counterfactual actions to take.</p>
      <p>The architectural space has been exhausted. Breaking the wall requires a fundamentally different kind of substrate: one with genuine closed-loop agency, where pattern actions cause environmental changes that pattern sensors then detect. This is not an incremental substrate modification but a change in the problem formulation.</p>
      </Section>

      <Section title="Falsification Map" level={1}>
      <table>
      <thead><tr><th>Experiment</th><th>Prediction</th><th>Outcome</th></tr></thead>
      <tbody>
      <tr><td>2 (World Model)</td><td><M>{"\\mathcal{C}_{\\text{wm}}"}</M> increases with evolution</td><td>Partially confirmed. Increases at bottleneck (100x), flat in general population. V15 re-run: memory helps (~12x on seed 42). V18 re-run: inconsistent improvement; bottleneck peak remains dominant.</td></tr>
      <tr><td>3 (Representation)</td><td>Compression and modeling co-emerge</td><td>Partially confirmed. Co-emerge under bottleneck (seed 123). Compression is cheap; quality improves only under selection.</td></tr>
      <tr><td>4 (Language)</td><td>Compositional communication emerges</td><td>Not confirmed. Chemical commons (MI {">"} 0) but <M>{"\\rho_{\\text{topo}} \\approx 0"}</M>. Unstructured broadcast, not language.</td></tr>
      <tr><td>5 (Counterfactual)</td><td>Reactive-to-detached transition</td><td>Null. Patterns always internally driven. Coupling wall. V15 re-run: wall persists. V18 re-run: wall persists despite boundary-gated architecture (<M>{"\\rho_{\\text{sync}} \\approx 0.003"}</M>). Definitively about agency, not signal routing.</td></tr>
      <tr><td>6 (Self-Model)</td><td>SM emergence with <M>{"\\intinfo"}</M> jump</td><td>Weak. <M>{"\\text{SM}_{\\text{sal}} > 1"}</M> once (n=1, bottleneck). No <M>{"\\intinfo"}</M> correlation. Coupling wall. V15 re-run: wall persists. V18 re-run: no improvement from boundary-dependent dynamics.</td></tr>
      <tr><td>7 (Affect Geometry)</td><td>Tripartite alignment</td><td>Partially confirmed. A-C develops over evolution (seed 7: 0.01 to 0.38). A-B null. V15 re-run: weaker (2/21 significant). V18 re-run: mixed (5/18 significant); higher robustness does not consistently improve geometry alignment.</td></tr>
      <tr><td>8 (<M>{"\\iota"}</M> Emergence)</td><td>Participatory default, animism</td><td><strong>Confirmed.</strong> <M>{"\\iota \\approx 0.30"}</M>, animism {">"} 1.0 in all 20 snapshots.</td></tr>
      <tr><td>9 (Normativity)</td><td>Viability gradient penalizes exploitation</td><td>Null. No <M>{"\\Delta V"}</M> asymmetry. Requires agency (capacity to act otherwise).</td></tr>
      <tr><td>10 (Superorganism)</td><td><M>{"\\intinfo_G > \\sum \\intinfo_i"}</M></td><td>Not confirmed. Ratio reaches 12% and increasing. No specialization.</td></tr>
      <tr><td>11 (Entanglement)</td><td>Co-emergence of modeling/abstraction/imagination</td><td>Not confirmed. Different cluster structure. Overall entanglement increases.</td></tr>
      <tr><td>12 (Capstone)</td><td>Seven criteria for identity thesis</td><td>All met (most at moderate/weak strength). Geometry confirmed; dynamics undertested.</td></tr>
      </tbody>
      </table>
      </Section>

      <Section title="Summary" level={1}>
      <p>The experimental program has three layers of results:</p>
      <ol>
      <li><strong>What was confirmed</strong>: Affect geometry is a baseline property of multi-agent survival (V10). Content-based coupling under lethal selection produces biological-like integration at population bottlenecks (V13). Temporal memory is the one substrate extension evolution consistently selects for (V15). Computational animism is the default perceptual mode (Experiment 8). Affect geometry develops over evolution and aligns with behavior (Experiment 7). Representation compression is cheap; quality improves under selection (Experiment 3).</li>
      <li><strong>What was not confirmed</strong>: Compositional communication (Experiment 4). Co-emergence of modeling/abstraction/imagination as predicted (Experiment 11). Superorganism integration (Experiment 10). Proto-normativity (Experiment 9).</li>
      <li><strong>What could not be tested</strong>: Three experiments (5, 6, 9) and re-runs (2, 7 on V15 and V18) hit the sensory-motor coupling wall. V18 definitively established that architectural approaches — signal routing changes, boundary gating, interior insulation — cannot break the wall. Breaking it requires a fundamentally different substrate with genuine closed-loop agency: patterns that act on the world and observe the consequences of their actions.</li>
      </ol>
      <p>V18 achieved the highest robustness of any substrate (mean 0.969, peak 1.651) while confirming that the coupling wall is about agency, not architecture. Seven experiments found positive signal. Three hit the coupling wall. One (Experiment 11) found that everything correlates with everything else but not in the clusters the theory predicted. The framework is not confirmed. It is informed.</p>
      </Section>
    </>
  );
}
