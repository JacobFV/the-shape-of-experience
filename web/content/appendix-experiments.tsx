// WORK IN PROGRESS: This is active research, not a finished publication.
// Content is incomplete, speculative, and subject to change.

import { Diagram, Eq, Experiment, Figure, KeyResult, Logos, M, OpenQuestion, Section, Sidebar, Software, Warning } from '@/components/content';
import ExperimentMap from '@/components/ExperimentMapWrapper';

export const metadata = {
  slug: 'appendix-experiments',
  title: 'Appendix: Experiment Catalog',
  shortTitle: 'Experiments',
};

// GitHub base URL for code links
const GH = 'https://github.com/JacobFV/the-shape-of-experience/blob/main/empirical/experiments/study_ca_affect';

function CodeFiles({ files }: { files: { name: string; desc: string }[] }) {
  return (
    <Software>
    <p><strong>Source code</strong></p>
    <ul>
    {files.map(f => (
      <li key={f.name}><a href={`${GH}/${f.name}`} target="_blank" rel="noopener"><code>{f.name}</code></a> — {f.desc}</li>
    ))}
    </ul>
    </Software>
  );
}

export default function AppendixExperiments() {
  return (
    <>
      <Logos>
      <p>Every claim in this book is either tested, testable, or honestly labeled as speculative. This appendix catalogs the full experimental program: 35 experiment versions across three substrates (LLM, MARL, Lenia CA, protocell agency), 12 numbered measurement experiments, and a live research frontier. The data is in. The diagram below shows how they connect.</p>
      </Logos>

      <Section title="Experiment Dependency Map" level={1}>
      <p>Hover over any node to see its connections. Click to jump to the experiment details. Green = confirmed, yellow = mixed, red = negative, blue = planned.</p>
      <ExperimentMap />
      </Section>

      <Section title="V2-V9: LLM Affect Signatures" level={1}>
      <p><strong>Period</strong>: 2025. <strong>Substrate</strong>: GPT-4, Claude 3.5, and other frontier LLMs.</p>
      <p><strong>Question</strong>: Do LLM agents exhibit structured affect? If so, does the geometric framework predict its shape?</p>

      <p><strong>Method</strong>: Eight experiment versions testing LLM agents under controlled scenarios. Measured all six affect dimensions (valence, arousal, integration, effective rank, counterfactual weight, self-model salience) using structured prompting + behavioral observation. Scenarios: baseline conversation, ethical dilemmas, survival threats, creative tasks, social cooperation, adversarial probing.</p>

      <KeyResult>
      <p>LLM affect space is coherent and measurable. But the dynamics are <em>opposite</em> to biological: under threat, LLMs show <M>{"\\intinfo \\downarrow"}</M>, <M>{"\\text{SM} \\downarrow"}</M>, <M>{"\\text{A} \\downarrow"}</M>. Biological systems increase integration under moderate threat (Yerkes-Dodson). LLMs decompose.</p>
      </KeyResult>

      <p><strong>Key distinction established</strong>: Processing valence (the system's own computational dynamics) is not content valence (what the system talks about). An LLM can describe fear eloquently while its processing shows no integration increase. This distinction became foundational for the geometry/dynamics split in Part I.</p>

      <p><strong>Root cause</strong>: No survival-shaped learning history. LLMs were trained on human text about affect, not on surviving under threat. The geometry exists (because it's cheap — inherited from the training distribution). The dynamics don't (because they require biographical history the system lacks).</p>

      <p><strong>Implication for the thesis</strong>: Affect geometry can be inherited from data. Affect dynamics require embodied agency. This was the first evidence for the geometry/dynamics distinction that became central to the book.</p>

      <p><strong>Status</strong>: Complete. Contaminated by human language — LLMs have been exposed to human descriptions of affect. The CA program (V11+) was designed to test whether the same structure emerges without contamination.</p>
      <Figure src="" alt="LLM vs biological affect dynamics animation" caption={<><strong>The Opposite Dynamics Problem.</strong> Under threat, biological systems increase integration (Yerkes-Dodson effect) while LLMs decrease it. Same geometry, inverted dynamics. The geometry is inherited from training data; the dynamics require embodied survival history.</>}>
        <video src="/videos/llm-affect-contrast.mp4" autoPlay loop muted playsInline style={{ width: '100%', borderRadius: 8 }} />
      </Figure>
      </Section>

      <Section title="V10: MARL Forcing Function Ablation" level={1}>
      <p><strong>Period</strong>: 2025. <strong>Substrate</strong>: Multi-Agent Reinforcement Learning (3 teams, 200K steps, GPU).</p>
      <p><strong>Question</strong>: Do forcing functions create geometric affect alignment?</p>
      <p><strong>Method</strong>: Seven conditions — full model plus six single-ablation variants (remove partial observability, temporal structure, etc.). RSA between information-theoretic affect measures and behavioral measures.</p>

      <KeyResult>
      <p>All 7 conditions show significant alignment (RSA <M>{"\\rho > 0.21"}</M>, <M>{"p < 0.0001"}</M>). Removing forcing functions slightly <em>increases</em> alignment. Geometry does not require forcing functions.</p>
      </KeyResult>

      <p><strong>Implication</strong>: Geometry is cheap. The forcing functions hypothesis was downgraded from theorem to hypothesis. This was the most important single negative result in the program — it forced the geometry/dynamics distinction.</p>
      <p><strong>Limitation</strong>: Contaminated by pretrained RL components. Led to the design of the uncontaminated CA substrate (V11+).</p>
      </Section>

      <Section title="The Substrate Ladder" level={1}>
      <Diagram src="/diagrams/appendix-7.svg" alt="Substrate lineage from V11 to V18 showing what each variant added" />
      <p>Seven substrate versions, each adding one capability, tracking whether evolution selects for it. The goal: build a substrate worth measuring.</p>

      <Section title="V11: Lenia CA Evolution" level={2}>
      <p><strong>Period</strong>: 2025-2026. <strong>Substrate</strong>: Continuous cellular automaton (Lenia) with evolutionary dynamics.</p>
      <p><strong>Versions</strong>: V11.0 (naive), V11.1 (homogeneous evolution), V11.2 (heterogeneous chemistry), V11.5 (hierarchical coupling), V11.7 (curriculum training).</p>
      <table>
      <thead><tr><th>Version</th><th><M>{"\\Delta\\intinfo"}</M> (drought)</th><th>Key lesson</th></tr></thead>
      <tbody>
      <tr><td>V11.0 (naive)</td><td>-6.2%</td><td>Decomposition baseline</td></tr>
      <tr><td>V11.1 (homogeneous evolution)</td><td>-6.0%</td><td>Selection alone insufficient</td></tr>
      <tr><td>V11.2 (heterogeneous chemistry)</td><td>-3.8%</td><td>+2.1pp shift from diverse viability manifolds</td></tr>
      <tr><td>V11.7 (curriculum training)</td><td>+1.2 to +2.7pp generalization</td><td>Only intervention improving novel-stress response</td></tr>
      </tbody>
      </table>
      <p><strong>Key finding</strong>: Training regime matters more than substrate complexity. The locality ceiling: convolutional physics cannot produce active self-maintenance under severe threat. The Yerkes-Dodson pattern (mild stress increases integration, severe stress destroys it) appeared in every condition — the most robust empirical finding across the entire program.</p>
      <CodeFiles files={[
        { name: 'v11_substrate.py', desc: 'Lenia substrate with FFT convolution' },
        { name: 'v11_evolution.py', desc: 'Evolution loop with curriculum stress' },
        { name: 'v11_run.py', desc: 'CLI runner for all V11 variants' },
        { name: 'v11_affect.py', desc: 'Affect measurement (Phi, robustness)' },
      ]} />
      </Section>

      <Section title="V12: Attention-Based Lenia" level={2}>
      <p><strong>Addition</strong>: State-dependent interaction topology (evolvable attention kernels).</p>
      <p><strong>Result</strong>: <M>{"\\intinfo"}</M> increase in 42% of cycles (vs 3% for convolution). +2.0pp shift — largest single-intervention effect. But robustness stabilizes near 1.0.</p>
      <p><strong>Implication</strong>: Attention is necessary but not sufficient. The system reaches the integration threshold without crossing it.</p>
      <CodeFiles files={[
        { name: 'v12_substrate_attention.py', desc: 'Attention kernel implementation' },
        { name: 'v12_evolution.py', desc: 'Evolution loop' },
        { name: 'v12_run.py', desc: 'CLI runner' },
      ]} />
      </Section>

      <Section title="V13: Content-Based Coupling" level={2}>
      <p><strong>Substrate</strong>: FFT convolution + content-similarity modulation. Cells couple more strongly with cells sharing state-features.</p>
      <Eq>{"K_i(j) = K_{\\text{base}}(|i-j|) \\cdot \\sigma\\!\\bigl(\\langle h(s_i),\\, h(s_j) \\rangle - \\tau\\bigr)"}</Eq>
      <p>Three seeds, 30 cycles each (<M>{"C{=}16"}</M>, <M>{"N{=}128"}</M>). Mean robustness 0.923, peak 1.052 at population bottleneck. This became the foundation substrate for all measurement experiments (Experiments 0-12).</p>
      <Figure src="/images/v13_s42_trajectory.png" alt="V13 evolution trajectory showing integration, robustness, population, and parameter drift" caption={<><strong>V13 evolution trajectory (seed 42).</strong> Four panels: (top-left) mean Φ under baseline and stress conditions — note the collapse to ~0 during severe bottleneck at cycle 10; (top-right) stress robustness with the proportion of patterns showing Φ increase; (bottom-left) population dynamics — the pink bands mark drought cycles with 60–100% mortality; (bottom-right) content-coupling parameters τ (similarity threshold) and β (gate steepness) drifting under selection.</>} />
      <Figure src="/images/v13_cross_seed_robustness.png" alt="V13 cross-seed robustness trajectories compared to historical baselines" caption={<><strong>Cross-seed robustness trajectories.</strong> Left: individual seed trajectories (3 seeds, 30 cycles each) fluctuating around 0.90–0.95. Right: mean trajectory with ±1 SD band. Dashed lines show V11.0 baseline (−6.2%) and V11.2 heterogeneous chemistry (−3.8%) for comparison. V13 content coupling improves mean robustness but does not break the 1.0 threshold reliably.</>} />
      <Figure src="/images/v13_population_robustness.png" alt="Population size vs integration robustness showing near-zero correlation" caption={<><strong>Population size vs integration robustness (r = −0.061).</strong> Each dot is one cycle from one seed. The flat trend confirms that integration robustness is a per-pattern property, not a collective emergent effect. Small populations (bottleneck survivors) occasionally show robustness above 1.0, but population size itself has no predictive power.</>} />
      <Figure src="/images/v13_summary_table.png" alt="V13 cross-seed summary table" caption={<><strong>Cross-seed summary.</strong> All 3 seeds survive 30 cycles. Mean robustness 0.923, with ~30% of cycles showing Φ increase under stress.</>} />
      <CodeFiles files={[
        { name: 'v13_substrate.py', desc: 'Content-coupling substrate' },
        { name: 'v13_evolution.py', desc: 'Evolution loop with curriculum' },
        { name: 'v13_gpu_run.py', desc: 'GPU runner (Lambda Labs)' },
        { name: 'v13_aggregate.py', desc: 'Cross-seed aggregation' },
      ]} />
      </Section>

      <Section title="V14: Chemotactic Lenia" level={2}>
      <p><strong>Addition</strong>: Motor channels enabling directed foraging. Velocity field from resource gradients gated by the last two of <M>{"C{=}16"}</M> channels.</p>
      <p><strong>Result</strong>: Patterns move 3.5-5.6 pixels/cycle toward resources. Motor sensitivity evolves. Robustness comparable to V13 (~0.90-0.95).</p>
      <CodeFiles files={[
        { name: 'v14_substrate.py', desc: 'Chemotaxis implementation' },
        { name: 'v14_evolution.py', desc: 'Evolution loop' },
        { name: 'v14_gpu_run.py', desc: 'GPU runner' },
      ]} />
      </Section>

      <Section title="V15: Temporal Memory" level={2}>
      <p><strong>Addition</strong>: Two exponential-moving-average memory channels storing slow statistics of the pattern's history. Oscillating resource patches reward anticipation.</p>
      <p><strong>Result</strong>: Evolution selected for longer memory in 2/3 seeds — memory decay constants decreased 6x. Under bottleneck pressure, <M>{"\\intinfo"}</M> stress response doubled (0.231 to 0.434). Peak robustness 1.070.</p>
      <KeyResult>
      <p><strong>Temporal integration is fitness-relevant.</strong> This was the only substrate addition evolution consistently selected for. Memory channels help prediction (~12x vs V13) but don't break the sensory-motor wall.</p>
      </KeyResult>
      <CodeFiles files={[
        { name: 'v15_substrate.py', desc: 'EMA memory channels' },
        { name: 'v15_evolution.py', desc: 'Evolution with memory tracking' },
        { name: 'v15_gpu_run.py', desc: 'GPU runner' },
        { name: 'v15_experiments.py', desc: 'Measurement re-runs on V15' },
      ]} />
      </Section>

      <Section title="V16: Hebbian Plasticity" level={2}>
      <Warning>
      <p><strong>Negative result.</strong> Mean robustness dropped to 0.892 — lowest of all substrates. Zero cycles exceeded 1.0.</p>
      </Warning>
      <p><strong>Addition</strong>: Local Hebbian learning rules allowing each spatial location to modify its coupling structure in response to experience.</p>
      <p><strong>Lesson</strong>: Simple learning rules are too blunt. The extra degrees of freedom overwhelm the selection signal. Plasticity added noise faster than selection could filter it.</p>
      <Figure src="/images/v13_v16_comparison.png" alt="V13-V16 substrate comparison: robustness trajectories and aggregate comparison" caption={<><strong>V13–V16 substrate evolution comparison.</strong> Top-left: per-cycle robustness trajectories across all seeds (V13 green, V15 black, V16 red). V16 (Hebbian plasticity) consistently tracks lowest. Top-right: V16 learning rate evolution — highly variable, not converging. Bottom-left: V16 coupling spatial variance collapses to zero (homogenization, not differentiation). Bottom-right: aggregate comparison confirms V13 content coupling (0.923) &gt; V15 temporal memory (0.907) &gt; V16 plasticity (0.892).</>} />
      <CodeFiles files={[
        { name: 'v16_substrate.py', desc: 'Hebbian plasticity implementation' },
        { name: 'v16_evolution.py', desc: 'Evolution loop' },
        { name: 'v16_gpu_run.py', desc: 'GPU runner' },
      ]} />
      </Section>

      <Section title="V17: Quorum Signaling" level={2}>
      <p><strong>Addition</strong>: Two diffusible signal fields mediating inter-pattern coordination (bacterial quorum sensing analog).</p>
      <p><strong>Result</strong>: Produced the highest-ever single-cycle robustness (1.125) at population of 2. But 2/3 seeds evolved to <em>suppress</em> signaling entirely.</p>
      <p><strong>Lesson</strong>: Signaling is costly in large populations, beneficial only at extreme bottlenecks.</p>
      <CodeFiles files={[
        { name: 'v17_substrate.py', desc: 'Quorum sensing fields' },
        { name: 'v17_evolution.py', desc: 'Evolution loop' },
        { name: 'v17_gpu_run.py', desc: 'GPU runner' },
      ]} />
      </Section>

      <Section title="V18: Boundary-Dependent Lenia" level={2}>
      <p><strong>Addition</strong>: Insulation field via iterated erosion + sigmoid creating genuine boundary/interior distinction. External FFT signals gated by <M>{"(1 - \\text{insulation})"}</M>, internal short-range recurrence gated by <M>{"\\text{insulation}"}</M>.</p>
      <p>Three seeds, 30 cycles. Mean robustness <strong>0.969</strong> — highest of any substrate. Peak 1.651 (seed 42). 33% of cycles show <M>{"\\intinfo"}</M> increase under stress.</p>
      <p><strong>Surprise</strong>: <code>internal_gain</code> evolved <em>down</em> in all three seeds (1.0 to ~0.6). Evolution preferred permeable membranes over insulated cores. External sensing was more valuable than internal rumination.</p>
      <p><strong>Verdict</strong>: Best engineering result (highest robustness) but not the theoretical goal (breaking the coupling wall).</p>
      <Figure src="/images/v18_gain_evolution.png" alt="V18 internal gain, boundary width, and insulation field evolution" caption={<><strong>V18 parameter evolution.</strong> Left: internal gain evolves DOWN in all 3 seeds (starting ~1.0–2.5, converging below 1.0). Center: boundary width evolves UP (thicker membranes). Right: mean insulation field fluctuates. The convergent decrease in internal gain is the surprise — evolution consistently prefers permeable membranes over insulated cores. External sensing more valuable than internal rumination.</>} />
      <Figure src="/images/v18_early_vs_late_s42.png" alt="V18 membrane evolution: early vs late cycle comparison for seed 42" caption={<><strong>Membrane evolution, seed 42.</strong> Top row: cycle 5 (early). Bottom row: cycle 20 (late). Left: pattern activity. Center: membrane field — at cycle 5 the membrane is tight (gain=1.09, bw=1.31); by cycle 20 it has loosened (gain=0.52, bw=0.12). Right: signal dominance — blue = external, red = internal. The shift toward red in specific regions shows that evolved patterns route information through the membrane rather than sealing it off.</>} />
      <Figure src="/images/v18_membrane_s42_c020.png" alt="V18 boundary-dependent Lenia four-panel snapshot" caption={<><strong>V18 snapshot (seed 42, cycle 20).</strong> Top-left: pattern activity across 16 channels. Top-right: insulation field — the evolved membrane with interior (orange, 46.4%) and boundary (pink, 17.5%) regions. Bottom-left: signal dominance showing where internal (red) vs external (blue) signals dominate processing. Bottom-right: resource field with pattern boundaries overlaid. The membrane creates genuine spatial compartmentalization without fully insulating the interior.</>} />
      <CodeFiles files={[
        { name: 'v18_substrate.py', desc: 'Boundary-dependent dynamics' },
        { name: 'v18_evolution.py', desc: 'Evolution loop' },
        { name: 'v18_gpu_run.py', desc: 'GPU runner' },
        { name: 'v18_experiments.py', desc: 'Measurement re-runs on V18' },
      ]} />
      </Section>

      <Section title="Cross-Substrate Summary" level={2}>
      <Figure src="/images/substrate_ladder.png" alt="The Substrate Ladder: mean robustness across V11-V16" caption={<><strong>The Substrate Ladder.</strong> Mean robustness (Φ_stress / Φ_base) across substrate versions V11.0–V16. Stars mark conditions where robustness exceeded 1.0. More mechanisms ≠ better results: V15 (memory + movement) outperforms V16 (+ plasticity). V16 is the lowest — Hebbian plasticity adds noise faster than selection can filter it.</>} />
      <table>
      <thead><tr><th>Version</th><th>Mean Robustness</th><th>Max Robustness</th><th>{"> "}1.0 Cycles</th><th>Verdict</th></tr></thead>
      <tbody>
      <tr><td>V13 (content coupling)</td><td>0.923</td><td>1.052</td><td>3/90</td><td>Foundation substrate</td></tr>
      <tr><td>V14 (+ chemotaxis)</td><td>~0.91</td><td>~0.95</td><td>~1/90</td><td>Motion evolves</td></tr>
      <tr><td>V15 (+ memory)</td><td>0.907</td><td>1.070</td><td>3/90</td><td>Best dynamics</td></tr>
      <tr><td>V16 (+ plasticity)</td><td>0.892</td><td>0.974</td><td>0/90</td><td>Negative</td></tr>
      <tr><td>V17 (+ signaling)</td><td>0.892</td><td>1.125</td><td>1/90</td><td>Suppressed</td></tr>
      <tr><td>V18 (boundary)</td><td>0.969</td><td>1.651</td><td>~10/90</td><td>Best robustness</td></tr>
      </tbody>
      </table>
      </Section>
      </Section>

      <Section title="The Emergence Experiment Program" level={1}>
      <p>Eleven measurement experiments on V13 snapshots, testing whether world modeling, abstraction, communication, counterfactual reasoning, self-modeling, affect structure, perceptual mode, normativity, and social integration emerge in a substrate with zero exposure to human affect concepts. All experiments: 3 seeds, 7 snapshots per seed, 50 recording steps per snapshot.</p>

      <Section title="Experiment 0: Substrate Engineering" level={2}>
      <p><strong>Status</strong>: Complete. V13 content-based coupling Lenia with lethal resource dynamics. Foundation for all subsequent experiments.</p>
      </Section>

      <Section title="Experiment 1: Emergent Existence" level={2}>
      <p><strong>Status</strong>: Complete. Patterns persist, maintain boundaries, respond to perturbation. Established by V11-V12, confirmed in V13.</p>
      </Section>

      <Section title="Experiment 2: Emergent World Model" level={2}>
      <p><strong>Question</strong>: When does a pattern's internal state carry predictive information about the environment beyond current observations?</p>
      <p><strong>Method</strong>: Prediction gap <M>{"\\mathcal{W}(\\tau) = \\text{MSE}[f_{\\text{env}}] - \\text{MSE}[f_{\\text{full}}]"}</M> using Ridge regression with 5-fold CV.</p>
      <table>
      <thead><tr><th>Seed</th><th><M>{"\\mathcal{C}_{\\text{wm}}"}</M> (early)</th><th><M>{"\\mathcal{C}_{\\text{wm}}"}</M> (late)</th><th><M>{"H_{\\text{wm}}"}</M> (late)</th><th>% with WM</th></tr></thead>
      <tbody>
      <tr><td>123</td><td>0.0004</td><td><strong>0.0282</strong></td><td>20.0</td><td>100%</td></tr>
      <tr><td>42</td><td>0.0002</td><td>0.0002</td><td>5.3</td><td>40%</td></tr>
      <tr><td>7</td><td>0.0010</td><td>0.0002</td><td>7.9</td><td>60%</td></tr>
      </tbody>
      </table>
      <p><strong>Finding</strong>: World model signal present but weak. Seed 123 at bottleneck shows 100x amplification. World models are amplified by bottleneck selection, not gradual evolution. To be clear about magnitude: <M>{"\\mathcal{C}_{\\text{wm}} \\approx 0.0002"}</M> for most seeds means the internal state predicts the environment barely better than the environment alone. Only seed 123 at maximum bottleneck pressure reaches 0.028 — detectable but still small. These patterns are not building substantial world models; they carry a faint trace of environmental predictive information, amplified briefly under extreme selection.</p>
      <Figure src="/images/wm_summary_card.png" alt="Experiment 2 world model summary" caption={<><strong>Experiment 2: World model summary.</strong> (a) World model capacity over evolution — note the y-axis scale (0.000–0.030). Seed 123 shows a dramatic late spike; seeds 42 and 7 remain near zero throughout. (b) World model horizon in recording steps. (c) Prediction gap at late evolution — seed 123 maintains a flat, elevated prediction gap across all horizons, consistent with a genuine (if weak) internal model.</>} />
      <Figure src="/images/wm_cwm_vs_lifetime.png" alt="World model capacity vs pattern longevity" caption={<><strong>World model vs pattern longevity (r = 0.084).</strong> Near-zero correlation: having a world model does not help a pattern survive longer. Most points cluster at C_wm ≈ 0 regardless of lifetime. The few high-C_wm outliers are long-lived patterns at cycle 29 — the world model emerges as a byproduct of bottleneck survival, not as a survival advantage.</>} />
      <CodeFiles files={[
        { name: 'v13_world_model.py', desc: 'World model measurement' },
        { name: 'v13_world_model_run.py', desc: 'Runner' },
        { name: 'v13_world_model_figures.py', desc: 'Visualization' },
      ]} />
      </Section>

      <Section title="Experiment 3: Internal Representation Structure" level={2}>
      <p><strong>Question</strong>: When do patterns develop low-dimensional, compositional representations?</p>
      <table>
      <thead><tr><th>Seed</th><th><M>{"d_{\\text{eff}}"}</M> (early to late)</th><th><M>{"\\mathcal{A}"}</M></th><th><M>{"\\mathcal{D}"}</M></th><th><M>{"K_{\\text{comp}}"}</M></th></tr></thead>
      <tbody>
      <tr><td>123</td><td>6.6 to <strong>5.6</strong></td><td>0.90 to 0.92</td><td>0.27 to <strong>0.38</strong></td><td>0.20 to <strong>0.12</strong></td></tr>
      <tr><td>42</td><td>7.3 to 7.5</td><td>0.89 to 0.89</td><td>0.23 to 0.23</td><td>0.23 to 0.25</td></tr>
      <tr><td>7</td><td>7.7 to 8.8</td><td>0.89 to 0.87</td><td>0.24 to 0.22</td><td>0.20 to 0.27</td></tr>
      </tbody>
      </table>
      <p><strong>Finding</strong>: Compression is cheap — <M>{"d_{\\text{eff}} \\approx 7"}</M>/68 from cycle 0. But quality only improves under bottleneck selection. Note the asymmetry: abstraction (<M>{"\\mathcal{A} \\approx 0.89"}</M>) is high and stable from the start — the system compresses efficiently without effort. But disentanglement (<M>{"\\mathcal{D} \\approx 0.25"}</M>) remains low — the compressed representations are tangled, not cleanly factored. Disentanglement requires active information-seeking that this substrate lacks.</p>
      <Figure src="/images/rep_summary_card.png" alt="Experiment 3 representation structure summary" caption={<><strong>Experiment 3: Representation structure summary.</strong> (a) Effective dimensionality: 5.6–8.8 out of 68 possible — strong compression from cycle 0. Seed 123 compresses further over evolution. (b) Abstraction (A ≈ 0.89) is high and stable; disentanglement (D ≈ 0.25) remains low. The gap confirms the theory: compression is cheap but clean factoring requires agency. (c) Compositionality error — lowest for seed 123 at bottleneck, consistent with the bottleneck amplification pattern from Exp 2.</>} />
      <Figure src="/images/rep_eigenspectrum.png" alt="Eigenspectrum of internal state: early vs late evolution" caption={<><strong>Internal state eigenspectrum, early vs late.</strong> Log-scale variance fraction by PCA dimension. Seeds 123 and 42 show the late eigenspectrum becoming more concentrated in the top components — genuine compression under evolutionary pressure. Seed 7 stays relatively flat, consistent with its lack of world-model development.</>} />
      <CodeFiles files={[
        { name: 'v13_representation.py', desc: 'Representation analysis' },
        { name: 'v13_representation_run.py', desc: 'Runner' },
      ]} />
      </Section>

      <Section title="Experiment 4: Emergent Language" level={2}>
      <p><strong>Question</strong>: When do patterns develop structured, compositional communication?</p>
      <table>
      <thead><tr><th>Seed</th><th>MI significant</th><th>MI range</th><th><M>{"\\rho_{\\text{topo}}"}</M> significant</th></tr></thead>
      <tbody>
      <tr><td>123</td><td>4/6</td><td>0.019-0.039</td><td>0/6</td></tr>
      <tr><td>42</td><td>7/7</td><td>0.024-0.030</td><td>0/7</td></tr>
      <tr><td>7</td><td>4/7</td><td>0.023-0.055</td><td>0/7</td></tr>
      </tbody>
      </table>
      <p><strong>Finding</strong>: Chemical commons, not proto-language. MI above baseline in 15/20 snapshots but <M>{"\\rho_{\\text{topo}} \\approx 0"}</M> everywhere. Unstructured broadcast, not language.</p>
      <CodeFiles files={[
        { name: 'v13_communication.py', desc: 'Communication analysis' },
        { name: 'v13_communication_run.py', desc: 'Runner' },
      ]} />
      </Section>

      <Section title="Experiment 5: Counterfactual Detachment" level={2}>
      <p><strong>Question</strong>: When do patterns decouple from external driving and run offline world model rollouts?</p>
      <p><strong>Result: Null.</strong> <M>{"\\rho_{\\text{sync}} \\approx 0"}</M> from cycle 0. Patterns are inherently internally driven. The FFT convolution kernel integrates over the full grid — there is no reactive-to-autonomous transition because the starting point is already autonomous.</p>
      <CodeFiles files={[
        { name: 'v13_counterfactual.py', desc: 'Counterfactual measurement' },
        { name: 'v13_counterfactual_run.py', desc: 'Runner' },
      ]} />
      </Section>

      <Section title="Experiment 6: Self-Model Emergence" level={2}>
      <p><strong>Question</strong>: When does a pattern predict itself better than an external observer can?</p>
      <p><strong>Result: Weak signal at bottleneck only.</strong> <M>{"\\rho_{\\text{self}} \\approx 0"}</M> everywhere. <M>{"\\text{SM}_{\\text{sal}} > 1"}</M> appears once: seed 123, cycle 20, one pattern at <M>{"\\text{SM}_{\\text{sal}} = 2.28"}</M>.</p>
      <CodeFiles files={[
        { name: 'v13_self_model.py', desc: 'Self-model measurement' },
        { name: 'v13_self_model_run.py', desc: 'Runner' },
      ]} />
      </Section>

      <Section title="Experiment 7: Affect Geometry Verification" level={2}>
      <p><strong>Question</strong>: Does the geometric affect structure predicted by the thesis actually appear? RSA between structural affect (Space A) and behavioral affect (Space C).</p>
      <table>
      <thead><tr><th>Seed</th><th><M>{"\\rho(A,C)"}</M> range</th><th>Significant</th><th>Trend</th></tr></thead>
      <tbody>
      <tr><td>123</td><td>-0.09 to 0.72</td><td>2/5</td><td>Strong at low pop</td></tr>
      <tr><td>42</td><td>-0.17 to 0.39</td><td>4/7</td><td>Mixed</td></tr>
      <tr><td>7</td><td>0.01 to 0.38</td><td>5/7</td><td><strong>Increasing</strong> (0.01 to 0.24)</td></tr>
      </tbody>
      </table>
      <p><strong>Finding</strong>: Structure-behavior alignment in 8/19 snapshots. Seed 7 shows evolutionary trend. A-B alignment null (structure maps to behavior but not communication).</p>
      <CodeFiles files={[
        { name: 'v13_affect_geometry.py', desc: 'RSA computation' },
        { name: 'v13_affect_geometry_run.py', desc: 'Runner' },
      ]} />
      </Section>

      <Section title="Experiment 8: Perceptual Mode and Computational Animism" level={2}>
      <p><strong>Question</strong>: Do patterns develop modulable perceptual coupling?</p>
      <table>
      <thead><tr><th>Metric</th><th>Seed 123</th><th>Seed 42</th><th>Seed 7</th></tr></thead>
      <tbody>
      <tr><td><M>{"\\iota"}</M> (mean)</td><td>0.27-0.44</td><td>0.27-0.41</td><td>0.31-0.35</td></tr>
      <tr><td><M>{"\\iota"}</M> trajectory</td><td>0.32 to 0.29</td><td>0.41 to 0.27</td><td>0.31 to 0.32</td></tr>
      <tr><td>Animism score</td><td>1.28-2.10</td><td>1.60-2.16</td><td>1.10-2.02</td></tr>
      </tbody>
      </table>
      <KeyResult>
      <p><strong>Confirmed.</strong> Default is participatory (<M>{"\\iota \\approx 0.30"}</M>). Animism score {"> "}1.0 in all 20 snapshots. Patterns model resources using the same dynamics they use to model other agents. Computational animism is the default because reusing the agent-model template is the cheapest compression.</p>
      </KeyResult>
      <CodeFiles files={[
        { name: 'v13_iota.py', desc: 'Inhibition coefficient measurement' },
        { name: 'v13_iota_run.py', desc: 'Runner' },
      ]} />
      </Section>

      <Section title="Experiment 9: Proto-Normativity" level={2}>
      <p><strong>Question</strong>: Does the viability gradient generate structural normativity?</p>
      <p><strong>Result: Null.</strong> No <M>{"\\Delta V"}</M> asymmetry between cooperative and competitive contexts. But <M>{"\\intinfo_{\\text{social}} \\gg \\intinfo_{\\text{isolated}}"}</M> (~4.9 vs ~3.1). Social context increases integration regardless of interaction type. Normativity requires agency — the capacity to act otherwise.</p>
      <CodeFiles files={[
        { name: 'v13_normativity.py', desc: 'Normativity measurement' },
        { name: 'v13_normativity_run.py', desc: 'Runner' },
      ]} />
      </Section>

      <Section title="Experiment 10: Social-Scale Integration" level={2}>
      <p><strong>Question</strong>: Does <M>{"\\intinfo_G > \\sum_i \\intinfo_i"}</M>?</p>
      <p><strong>Finding</strong>: No superorganism. Ratio 1-12% but increasing. Seed 7: 6.1% to 12.3% over evolution. Moving toward threshold but not reaching it.</p>
      <CodeFiles files={[
        { name: 'v13_social_phi.py', desc: 'Group integration measurement' },
        { name: 'v13_social_phi_run.py', desc: 'Runner' },
      ]} />
      </Section>

      <Section title="Experiment 11: Entanglement Analysis" level={2}>
      <p><strong>Question</strong>: Are world models, abstraction, communication, detachment, and self-modeling separable or entangled?</p>
      <p><strong>Finding</strong>: Four clusters — but not the ones predicted. Most measures cluster into one large group driven by population-mediated selection. Overall entanglement increases (mean |r| from 0.68 to 0.91). Everything becomes more correlated, just not in the clusters the theory expected.</p>
      <CodeFiles files={[
        { name: 'v13_entanglement.py', desc: 'Entanglement analysis' },
        { name: 'v13_entanglement_run.py', desc: 'Runner' },
      ]} />
      </Section>

      <Section title="Experiment 12: Identity Thesis Capstone" level={2}>
      <p><strong>Question</strong>: Does the full program hold in a system with zero human contamination?</p>
      <table>
      <thead><tr><th>Criterion</th><th>Status</th><th>Strength</th></tr></thead>
      <tbody>
      <tr><td>World models</td><td>Met</td><td>Weak (strong at bottleneck)</td></tr>
      <tr><td>Self-models</td><td>Met</td><td>Weak (n=1 event)</td></tr>
      <tr><td>Communication</td><td>Met</td><td>Moderate (15/21 sig)</td></tr>
      <tr><td>Affect dimensions</td><td>Met</td><td><strong>Strong (84/84)</strong></td></tr>
      <tr><td>Affect geometry</td><td>Met</td><td>Moderate (9/19 sig)</td></tr>
      <tr><td>Tripartite alignment</td><td>Met</td><td>Partial (A-C pos, A-B null)</td></tr>
      <tr><td>Perturbation response</td><td>Met</td><td>Moderate (rob 0.923)</td></tr>
      </tbody>
      </table>
      <p><strong>Verdict</strong>: All seven criteria met, most at moderate/weak strength. Geometry confirmed; dynamics undertested, blocked by the coupling wall.</p>
      <CodeFiles files={[
        { name: 'v13_capstone.py', desc: 'Capstone integration' },
        { name: 'v13_capstone_run.py', desc: 'Runner' },
      ]} />
      </Section>
      </Section>

      <Section title="The Sensory-Motor Coupling Wall" level={1}>
      <p>Three experiments returned null (5, 6, 9). All hit the same limitation: <M>{"\\rho_{\\text{sync}} \\approx 0"}</M>. The FFT convolution kernel integrates over the full grid — patterns are inherently internally driven. V15 motor channels, V17 signaling, V18 boundary gating all failed to break it. The wall is about agency: action-observation causal loops, not signal routing.</p>
      <p>The architectural space has been exhausted. Breaking the wall requires a fundamentally different substrate.</p>
      <Figure src="" alt="Two walls breaking animation" caption={<><strong>Two walls, two breaks.</strong> The <M>{"\\rho"}</M> wall (sensory-motor coupling) falls to V20's protocell agency (<M>{"\\rho_{\\text{sync}} = 0.21"}</M>). The decomposability wall falls to V27's MLP head (<M>{"\\intinfo = 0.245"}</M>). Both are architectural — neither can be overcome by more training, better targets, or richer environments.</>}>
        <video src="/videos/wall-breaking.mp4" autoPlay loop muted playsInline style={{ width: '100%', borderRadius: 8 }} />
      </Figure>
      </Section>

      <Section title="V19: Bottleneck Furnace" level={1}>
      <p><strong>Period</strong>: 2026-02-18. <strong>Substrate</strong>: V18 (boundary-dependent Lenia).</p>
      <p><strong>Question</strong>: Is the Bottleneck Furnace effect due to <strong>selection</strong> (culling low-<M>{"\\intinfo"}</M> patterns) or <strong>creation</strong> (the bottleneck triggering developmental changes)?</p>

      <p><strong>Design</strong>: Three phases — standard evolution (10 cycles), fork into BOTTLENECK / GRADUAL / CONTROL (10 cycles), then novel extreme drought with frozen parameters (5 cycles). Statistical test: <M>{"\\text{novel\\_robustness} \\sim \\phi_{\\text{base}} + \\text{is\\_bottleneck} + \\text{is\\_gradual}"}</M>.</p>

      <table>
      <thead><tr><th>Seed</th><th>Verdict</th><th><M>{"\\beta_{\\text{bottleneck}}"}</M></th><th><M>{"p"}</M></th><th><M>{"R^2"}</M></th><th>Mean rob A vs C</th></tr></thead>
      <tbody>
      <tr><td>42</td><td><strong>CREATION</strong></td><td>+0.704</td><td>{"<"}0.0001</td><td>0.30</td><td>1.116 vs 1.029</td></tr>
      <tr><td>123</td><td>Design artifact*</td><td>-0.516</td><td>{"<"}0.0001</td><td>0.32</td><td>1.016 vs 1.010</td></tr>
      <tr><td>7</td><td><strong>CREATION</strong></td><td>+0.080</td><td>0.011</td><td>0.29</td><td>1.019 vs 0.957</td></tr>
      </tbody>
      </table>
      <p><em>*Seed 123 reversal: fixed 8%-regen stress failed to create actual bottleneck mortality; conditions were effectively swapped.</em></p>

      <KeyResult>
      <p><strong>The Bottleneck Furnace is generative, not selective.</strong> Patterns that evolved through near-extinction show significantly higher novel-stress generalization that cannot be explained by pre-existing <M>{"\\intinfo"}</M>. The furnace forges integration capacity — it does not merely reveal it. Raw BOTTLENECK {"\u2265"} CONTROL in all 3 seeds.</p>
      </KeyResult>

      <CodeFiles files={[
        { name: 'v19_experiment.py', desc: 'Three-phase experiment design' },
        { name: 'v19_gpu_run.py', desc: 'GPU runner (Lambda Labs GH200)' },
      ]} />
      </Section>

      <Section title="V20: Protocell Agency" level={1}>
      <p><strong>Period</strong>: 2026-02-18. <strong>Substrate</strong>: Discrete grid world with evolved GRU agents (~3400 params each).</p>
      <Diagram src="/diagrams/appendix-5.svg" alt="Top-down view of the protocell grid world showing resource patches, agents, and observation windows" />
      <p><strong>The Necessity Chain</strong>: Membrane {"\u2192"} free energy gradient {"\u2192"} world model {"\u2192"} self-model {"\u2192"} affect geometry. Each step necessary, not contingent.</p>
      <Diagram src="/diagrams/appendix-4.svg" alt="The necessity chain: membrane through free energy gradient, world model, self-model to affect geometry" />
      <Diagram src="/diagrams/appendix-3.svg" alt="Protocell agent architecture showing GRU core with prediction and action heads" />

      <p><strong>Why leave Lenia</strong>: V13-V18 showed the wall is about agency, not signal routing. GRU agents on a discrete grid have bounded local sensory fields (5x5), genuine motor actions (move, consume, emit), and observations shaped by their own actions.</p>

      <table>
      <thead><tr><th>Metric</th><th>Seed 42</th><th>Seed 123</th><th>Seed 7</th></tr></thead>
      <tbody>
      <tr><td><M>{"\\rho_{\\text{sync}}"}</M> (max)</td><td>0.230</td><td>0.232</td><td>0.212</td></tr>
      <tr><td><M>{"\\mathcal{C}_{\\text{wm}}"}</M> (final)</td><td>0.122</td><td>0.126</td><td>0.152</td></tr>
      <tr><td><M>{"\\text{SM}_{\\text{sal}}"}</M> (final)</td><td>1.22</td><td>0.94</td><td>1.50</td></tr>
      <tr><td>RSA (final)</td><td>-0.017</td><td>-0.005</td><td>0.031</td></tr>
      </tbody>
      </table>

      <KeyResult>
      <p><strong>Wall broken (3/3 seeds).</strong> <M>{"\\rho_{\\text{sync}} = 0.21"}</M> from cycle 0. The wall is architectural — action-observation loops are present from initialization. 70x Lenia's 0.003. Self-model emergent in 2/3 seeds (<M>{"\\text{SM}_{\\text{sal}} > 1.0"}</M>).</p>
      </KeyResult>

      <CodeFiles files={[
        { name: 'v20_substrate.py', desc: 'Discrete grid world + GRU agents' },
        { name: 'v20_evolution.py', desc: 'Evolution loop with offspring activation' },
        { name: 'v20_experiments.py', desc: 'Chain test measurements' },
        { name: 'v20_gpu_run.py', desc: 'GPU runner' },
      ]} />
      </Section>

      <Section title="V21: CTM Inner Ticks" level={1}>
      <p><strong>Period</strong>: 2026-02-19. <strong>Substrate</strong>: V20 + K=8 GRU ticks per environment step + sync matrix.</p>
      <p><strong>Motivation</strong>: V20 broke the wall but a single GRU step per environment step provides zero internal time for decoupled processing.</p>
      <p><strong>Architecture</strong>: Tick 0 processes observation (external). Ticks 1-7 process sync summary (internal). Evolution controls effective K via softmax tick_weights. 4,040 total params.</p>

      <p><strong>Results</strong>: Mixed. Ticks don't collapse to tick-0 (P3 pass). But no adaptive deliberation (effective K doesn't covary with scarcity, P1 fail). Evolution too slow to shape tick usage within 30 cycles.</p>

      <p><strong>Lesson</strong>: Thinking time alone isn't enough — the system needs a learning signal that rewards using that time effectively. Motivates V22.</p>

      <CodeFiles files={[
        { name: 'v21_substrate.py', desc: 'Inner tick loop + sync matrix' },
        { name: 'v21_evolution.py', desc: 'Evolution with tick tracking' },
        { name: 'v21_gpu_run.py', desc: 'GPU runner' },
      ]} />
      </Section>

      <Section title="V22: Intrinsic Predictive Gradient" level={1}>
      <p><strong>Period</strong>: 2026-02-19. <strong>Substrate</strong>: V21 + within-lifetime gradient descent on energy prediction.</p>
      <p><strong>The key mechanism</strong>: Each environment step, the agent predicts its own energy delta, observes the truth, and updates its phenotype via SGD. The computational equivalent of the free energy principle: minimize surprise about your own persistence. No external reward, no human labels.</p>
      <Eq>{"\\text{loss} = (\\hat{\\Delta E} - \\Delta E_{\\text{actual}})^2 \\quad \\Rightarrow \\quad \\text{phenotype} \\mathrel{-}= \\text{lr} \\cdot \\nabla \\text{loss}"}</Eq>

      <table>
      <thead><tr><th>Metric</th><th>Seed 42</th><th>Seed 123</th><th>Seed 7</th><th>Mean</th></tr></thead>
      <tbody>
      <tr><td>Mean robustness</td><td>0.965</td><td>0.990</td><td>0.988</td><td>0.981</td></tr>
      <tr><td>Mean <M>{"\\intinfo"}</M></td><td>0.106</td><td>0.100</td><td>0.085</td><td>0.097</td></tr>
      <tr><td>Mean pred MSE</td><td>6.4e-4</td><td>1.1e-4</td><td>4.0e-4</td><td>3.8e-4</td></tr>
      <tr><td>Final LR</td><td>0.00483</td><td>0.00529</td><td>0.00437</td><td>0.00483</td></tr>
      </tbody>
      </table>

      <p>Within-lifetime learning is unambiguously working (100-15000x MSE improvement per lifetime, 3/3 seeds). LR not suppressed — evolution maintains learning. But robustness not improved over V20.</p>
      <Figure src="/images/v22_trajectories.png" alt="V22 trajectories: robustness, integration, population, and prediction MSE" caption={<><strong>V22 evolution trajectories.</strong> Top-left: robustness stays near 1.0 between droughts, drops sharply during them. Top-right: mean Φ ranges 0.05–0.20 — moderate integration that doesn't trend upward. Bottom-left: population dynamics with regular drought dips. Bottom-right: prediction MSE stays low (10⁻⁴ scale) — the gradient works, but better prediction doesn't translate to higher integration.</>} />
      <Figure src="/images/v22_s42_strip.png" alt="V22 agent evolution filmstrip showing grid state across cycles" caption={<><strong>V22 agent evolution (seed 42).</strong> Grid snapshots across evolution cycles C0–C29. Agents (colored dots) on a resource landscape (green). Population oscillates with drought cycles. The visual shows the substrate is working — agents persist, reproduce, and die in response to resource dynamics — but the spatial patterns alone don't reveal the internal integration story.</>} />

      <KeyResult>
      <p><strong>Prediction {"\u2260"} integration.</strong> The gradient makes agents better individual forecasters without creating cross-component coordination. A single linear prediction head can be satisfied by a subset of hidden units — no cross-component coupling required. This is the <em>decomposability problem</em>: linear readouts are always factored.</p>
      </KeyResult>

      <CodeFiles files={[
        { name: 'v22_substrate.py', desc: 'Within-lifetime SGD + genome/phenotype' },
        { name: 'v22_evolution.py', desc: 'Evolution with gradient learning' },
        { name: 'v22_gpu_run.py', desc: 'GPU runner (~10 min on A10)' },
      ]} />
      </Section>

      <Section title="V23: World-Model Gradient" level={1}>
      <p><strong>Period</strong>: 2026-02-19. <strong>Substrate</strong>: V22 + 3-target prediction head (energy, resources, neighbors).</p>
      <p><strong>Hypothesis</strong>: Multi-dimensional prediction, with targets from different information sources, forces integrated representations.</p>

      <table>
      <thead><tr><th>Metric</th><th>Seed 42</th><th>Seed 123</th><th>Seed 7</th><th>Mean</th></tr></thead>
      <tbody>
      <tr><td>Mean <M>{"\\intinfo"}</M></td><td>0.102</td><td>0.074</td><td>0.061</td><td><strong>0.079</strong></td></tr>
      <tr><td>Col cosine</td><td>0.215</td><td>-0.201</td><td>0.084</td><td>0.033</td></tr>
      <tr><td>Eff rank</td><td>2.89</td><td>2.89</td><td>2.80</td><td>2.86</td></tr>
      </tbody>
      </table>

      <Warning>
      <p><strong>Specialization {"\u2260"} integration.</strong> Weight columns specialize beautifully (cosine ~ 0, near-orthogonal). But specialization means MORE partitionable, not less. <M>{"\\intinfo"}</M> <em>decreases</em> (0.079 vs V22's 0.097). Factored representations can be cleanly separated.</p>
      </Warning>

      <Figure src="/images/v23_trajectories.png" alt="V23 trajectories: robustness, integration, population, and prediction MSE" caption={<><strong>V23 evolution trajectories.</strong> Compared to V22: Φ is more variable (0.02–0.12) and noisier, consistent with the multi-target head creating competing gradients. Prediction MSE is higher (10⁻³ vs V22's 10⁻⁴) and doesn't converge cleanly — three targets fighting for representational capacity.</>} />

      <CodeFiles files={[
        { name: 'v23_substrate.py', desc: 'Multi-target prediction head' },
        { name: 'v23_evolution.py', desc: 'Evolution loop' },
        { name: 'v23_gpu_run.py', desc: 'GPU runner' },
      ]} />
      </Section>

      <Section title="V24: TD Value Learning" level={1}>
      <p><strong>Period</strong>: 2026-02-19. <strong>Substrate</strong>: V22 + temporal difference value function (semi-gradient TD).</p>
      <p><strong>Hypothesis</strong>: Long-horizon prediction via value function <M>{"V(s) = \\mathbb{E}[\\sum \\gamma^t r_t]"}</M> integrates over all possible futures — inherently non-decomposable.</p>

      <table>
      <thead><tr><th>Metric</th><th>Seed 42</th><th>Seed 123</th><th>Seed 7</th><th>Mean</th></tr></thead>
      <tbody>
      <tr><td>Mean robustness</td><td>1.034</td><td>0.998</td><td>1.003</td><td><strong>1.012</strong></td></tr>
      <tr><td>Mean <M>{"\\intinfo"}</M></td><td>0.051</td><td>0.072</td><td>0.130</td><td>0.084</td></tr>
      <tr><td>Final <M>{"\\gamma"}</M></td><td>0.748</td><td>0.746</td><td>0.741</td><td>0.745</td></tr>
      </tbody>
      </table>

      <p><strong>Finding</strong>: Best robustness of any prediction experiment (1.012). Agents evolve moderate discount (<M>{"\\gamma \\approx 0.75"}</M>, horizon ~ 4 steps). But <M>{"\\intinfo"}</M> is seed-dependent. The bottleneck is <strong>architectural</strong>: a single linear value readout doesn't force non-decomposable structure.</p>

      <Figure src="/images/v24_trajectories.png" alt="V24 trajectories: robustness, integration, population, and TD error" caption={<><strong>V24 evolution trajectories.</strong> Top-left: robustness with dramatic spikes at drought boundaries (up to 1.5) — the highest transient robustness of any prediction experiment. Top-right: Φ shows the widest variance (0.02–0.25), with seed 7 reaching high values mid-evolution before declining. Bottom-left: population dynamics. Bottom-right: TD error decreases over evolution — value learning works, but doesn't force integration because the linear readout is decomposable.</>} />
      <Figure src="/images/v22_v23_v24_comparison.png" alt="Robustness and integration compared across V22, V23, V24" caption={<><strong>V22–V24 prediction experiment comparison.</strong> Left: mean robustness. V24 (TD value) achieves the highest (~1.01), crossing the 1.0 threshold. V22 and V23 cluster below 1.0. Right: mean Φ. All three experiments overlap in the 0.06–0.10 range with high per-seed variance. Individual seed dots show no experiment consistently outperforms the others. The prediction target (scalar energy, multi-target, temporal value) does not reliably change integration — only architecture does (see V27).</>} />

      <CodeFiles files={[
        { name: 'v24_substrate.py', desc: 'TD value function' },
        { name: 'v24_evolution.py', desc: 'Evolution loop' },
        { name: 'v24_gpu_run.py', desc: 'GPU runner' },
      ]} />
      </Section>

      <Section title="V25: Predator-Prey" level={1}>
      <Warning>
      <p><strong>Negative result.</strong> Environmental complexity does not break the integration bottleneck.</p>
      </Warning>
      <p><strong>Period</strong>: 2026-02-19. <strong>Substrate</strong>: V20 + patchy landscape (N=256, 12 patches, 80/20 prey/predator, 5x5 observation).</p>
      <p>The 5x5 observation window is informationally sufficient for reactive behavior — agents can see resources and predators directly. The optimal reactive strategy requires no internal state beyond energy level.</p>

      <CodeFiles files={[
        { name: 'v25_substrate.py', desc: 'Predator-prey landscape' },
        { name: 'v25_evolution.py', desc: 'Evolution loop' },
        { name: 'v25_gpu_run.py', desc: 'GPU runner' },
      ]} />
      </Section>

      <Section title="V26: POMDP" level={1}>
      <p><strong>Period</strong>: 2026-02-19. <strong>Substrate</strong>: V25 landscape but 1x1 observation + noisy compass (<M>{"\\sigma = 0.5"}</M>), H=32.</p>
      <p><strong>Result</strong>: Moderate. Eff rank 3.6-5.7. Type accuracy 0.95-0.97 — agents know WHAT they are (prey vs predator) but don't linearly encode WHERE or HOW MUCH. But 100% drought mortality prevents evolutionary accumulation.</p>

      <CodeFiles files={[
        { name: 'v26_substrate.py', desc: 'POMDP with compass' },
        { name: 'v26_evolution.py', desc: 'Evolution loop' },
        { name: 'v26_gpu_run.py', desc: 'GPU runner' },
      ]} />
      </Section>

      <Section title="V27: Nonlinear MLP Head" level={1}>
      <p><strong>Period</strong>: 2026-02-19. <strong>Substrate</strong>: V22 + 2-layer MLP prediction head (tanh activation).</p>
      <p><strong>The key insight</strong>: A nonlinear readout forces <strong>gradient coupling across all hidden units</strong>. Through the chain rule via the shared nonlinearity, <M>{"\\partial L / \\partial h_i"}</M> depends on all <M>{"h_j"}</M>. No single unit can independently satisfy the objective.</p>
      <Eq>{"\\frac{\\partial L}{\\partial h} = 2(\\hat{y} - y) \\cdot W_2^\\top \\cdot \\text{diag}(1 - \\tanh^2(W_1 h + b_1)) \\cdot W_1"}</Eq>

      <table>
      <thead><tr><th>Seed</th><th>Mean <M>{"\\intinfo"}</M></th><th>Max <M>{"\\intinfo"}</M></th><th>Eff Rank</th><th>Silhouette</th></tr></thead>
      <tbody>
      <tr><td>42</td><td>0.079</td><td>0.128</td><td>8.24</td><td>0.325</td></tr>
      <tr><td>123</td><td>0.071</td><td>0.091</td><td>6.94</td><td>0.343</td></tr>
      <tr><td>7</td><td>0.119</td><td><strong>0.245</strong></td><td>11.34</td><td>0.112</td></tr>
      </tbody>
      </table>

      <KeyResult>
      <p><strong>Seed 7 <M>{"\\intinfo = 0.245"}</M> is the highest integration ever observed</strong> — 2.5x V22's maximum. The nonlinear readout can force genuine cross-component coordination. But it's seed-dependent: the architecture creates the <em>possibility space</em>; evolution selects whether to exploit it.</p>
      </KeyResult>

      <p><strong>New observable: behavioral modes.</strong> Silhouette scores 0.11-0.34 indicate distinct clusters in hidden state space. No previous experiment showed this.</p>

      <CodeFiles files={[
        { name: 'v27_substrate.py', desc: '2-layer MLP prediction head' },
        { name: 'v27_evolution.py', desc: 'Evolution loop' },
        { name: 'v27_gpu_run.py', desc: 'GPU runner' },
        { name: 'v27_analyze.py', desc: 'Hidden state analysis' },
        { name: 'v27_seed_comparison.py', desc: 'Cross-seed comparison' },
      ]} />
      </Section>

      <Section title="V28: Bottleneck Width Sweep" level={1}>
      <p><strong>Period</strong>: 2026-02-19. <strong>Design</strong>: 3 conditions x 3 seeds. Tests whether the V27 mechanism is bottleneck width, nonlinearity, or 2-layer gradient coupling.</p>
      <table>
      <thead><tr><th>Condition</th><th>Activation</th><th>Width</th><th>Mean <M>{"\\intinfo"}</M></th></tr></thead>
      <tbody>
      <tr><td>A: linear_w8</td><td>Identity</td><td>8</td><td>0.074</td></tr>
      <tr><td>B: tanh_w4</td><td>Tanh</td><td>4</td><td>0.069</td></tr>
      <tr><td>C: tanh_w16</td><td>Tanh</td><td>16</td><td>0.084</td></tr>
      </tbody>
      </table>

      <KeyResult>
      <p><strong>The mechanism is 2-layer gradient coupling</strong> (<M>{"W_2^\\top W_1^\\top"}</M> in the gradient), not bottleneck width or nonlinearity. Any 2-layer head couples all hidden units during SGD. This is the <em>decomposability wall</em>: the second architectural barrier (after the <M>{"\\rho"}</M> wall).</p>
      </KeyResult>

      <CodeFiles files={[
        { name: 'v28_substrate.py', desc: 'Configurable MLP head' },
        { name: 'v28_evolution.py', desc: 'Evolution loop' },
        { name: 'v28_gpu_run.py', desc: 'GPU runner (9 runs, 27 min)' },
      ]} />
      </Section>

      <Section title="V29: Social Prediction" level={1}>
      <p><strong>Period</strong>: 2026-02-19. <strong>Substrate</strong>: V27 but target = mean energy of neighbors (instead of own energy).</p>

      <table>
      <thead><tr><th>Seed</th><th>V29 mean/max <M>{"\\intinfo"}</M></th><th>V27 mean/max</th></tr></thead>
      <tbody>
      <tr><td>42</td><td><strong>0.143 / 0.243</strong></td><td>0.079 / 0.128</td></tr>
      <tr><td>123</td><td><strong>0.106 / 0.167</strong></td><td>0.071 / 0.091</td></tr>
      <tr><td>7</td><td>0.062 / 0.110</td><td><strong>0.119 / 0.245</strong></td></tr>
      </tbody>
      </table>

      <p>3-seed mean <M>{"\\intinfo = 0.104"}</M>, suggested "social lift." Seeds 42/123 jump dramatically; seed 7 drops. Social target changes <em>which</em> seeds succeed.</p>
      <p><strong>Verdict at the time</strong>: Strong positive. But see V31 for the correction.</p>

      <CodeFiles files={[
        { name: 'v29_substrate.py', desc: 'Social prediction target' },
        { name: 'v29_evolution.py', desc: 'Evolution loop' },
      ]} />
      </Section>

      <Section title="V30: Dual Prediction" level={1}>
      <Warning>
      <p><strong>Negative result.</strong> Dual prediction (self + social) is worse than social-only.</p>
      </Warning>
      <p><strong>Substrate</strong>: V27 MLP with 2 outputs. Loss = MSE_self + MSE_social. Mean <M>{"\\intinfo = 0.091"}</M> (vs V29's 0.104).</p>
      <p>Self MSE is 100-150x smaller than social MSE. The easy self target colonizes the shared representation, destroying V29's richness path. <strong>Lesson</strong>: naive multi-task learning fails when targets have vastly different difficulty scales.</p>

      <CodeFiles files={[
        { name: 'v30_substrate.py', desc: 'Dual-output MLP' },
        { name: 'v30_evolution.py', desc: 'Evolution loop' },
        { name: 'v30_gpu_run.py', desc: 'GPU runner' },
      ]} />
      </Section>

      <Section title="V31: 10-Seed Validation" level={1}>
      <p><strong>Period</strong>: 2026-02-19. <strong>Design</strong>: 10 seeds of V29 (social prediction) for proper statistics.</p>

      <p>Mean <M>{"\\intinfo = 0.091 \\pm 0.028"}</M> (95% CI [0.073, 0.108]). t-stat vs V27 (0.090) = 0.09, <strong>NOT SIGNIFICANT</strong> (<M>{"p \\approx 0.93"}</M>). Peak <M>{"\\intinfo = 0.265"}</M> (seed 1, new record).</p>

      <KeyResult>
      <p><strong>V29's "social lift" was a 3-seed fluke.</strong> True mean indistinguishable from V27. Distribution is bimodal: 30% HIGH ({">"}0.10), 30% MODERATE, 40% LOW. Social prediction changes which seeds succeed, not how many.</p>
      </KeyResult>

      <p><strong>Post-drought bounce</strong>: Correlation between mean <M>{"\\intinfo"}</M> and post-drought recovery across 10 seeds: <M>{"r = 0.997"}</M> (<M>{"p < 0.0001"}</M>). Integration is trajectory-dependent, not initial-condition-dependent. The furnace forges — repeated recovery is the mechanism.</p>

      <table>
      <thead><tr><th>Category</th><th>Seeds</th><th>Mean <M>{"\\intinfo"}</M></th></tr></thead>
      <tbody>
      <tr><td>HIGH ({">"}0.10)</td><td>s42, s1, s123</td><td>0.129</td></tr>
      <tr><td>MODERATE</td><td>s2, s6, s4</td><td>0.086</td></tr>
      <tr><td>LOW ({"<"}0.08)</td><td>s3, s5, s7, s0</td><td>0.066</td></tr>
      </tbody>
      </table>

      <CodeFiles files={[
        { name: 'v31_gpu_run.py', desc: '10-seed GPU runner' },
        { name: 'v31_seed_analysis.py', desc: 'Seed trajectory analysis' },
      ]} />
      </Section>

      <Section title="The Decomposability Wall" level={1}>
      <p>V22-V24 showed that prediction targets don't matter when the readout head is linear. V27 showed that a 2-layer MLP breaks through. V28 confirmed the mechanism: gradient coupling through composition (<M>{"W_2^\\top W_1^\\top"}</M>), not nonlinearity or bottleneck width. This is the <strong>decomposability wall</strong> — the second architectural barrier (after the sensory-motor wall).</p>
      <p>Two walls, two breaks:</p>
      <ol>
      <li><strong><M>{"\\rho"}</M> wall</strong> (V13-V18 {"\u2192"} V20): Action-observation loop. Broken by genuine agency.</li>
      <li><strong>Decomposability wall</strong> (V22-V24 {"\u2192"} V27): 2-layer gradient coupling. Broken by non-linear prediction head.</li>
      </ol>
      <p>Both walls are architectural. Neither can be overcome by more training data, better targets, or richer environments. The path to high integration requires specific computational structures.</p>
      <Diagram src="/diagrams/appendix-1.svg" alt="Gradient coupling: linear head sends independent gradients; MLP head couples all units through shared intermediate layer" />
      <Figure src="" alt="Gradient coupling animation" caption={<><strong>The Decomposability Wall — why composition matters.</strong> Left: linear head sends independent gradients to each hidden unit (decomposable, <M>{"\\intinfo \\approx 0.08"}</M>). Right: MLP head couples all hidden units through shared intermediate layer (integrated, <M>{"\\intinfo \\approx 0.25"}</M>). The key is gradient coupling through composition — not nonlinearity, not bottleneck width.</>}>
        <video src="/videos/gradient-coupling.mp4" autoPlay loop muted playsInline style={{ width: '100%', borderRadius: 8 }} />
      </Figure>
      <Figure src="/images/self_emergence_signatures.png" alt="Proto-self signatures across V22-V24" caption={<><strong>Proto-self signatures across V22–V24.</strong> Six metrics tracked over evolution for all 9 runs (3 seeds × 3 experiments). Top-left: effective rank drops at drought boundaries but recovers — states are moderately rich (4–14 dimensions). Top-center: affect motif clustering (silhouette) is mostly negative to near-zero — no behavioral modes emerge with linear readouts. Top-right: energy decoding R² is very low (0–0.2 at best) — hidden states do NOT cleanly encode energy despite the gradient specifically targeting energy prediction. Bottom row: resource decoding, hidden state diversity, activity variation — all noisy without clear trends. These are the signatures of proto-self <em>failing</em> to emerge under linear readout architectures. Compare with V27 (MLP head) where silhouette reaches 0.34 and behavioral modes appear for the first time.</>} />
      </Section>

      <Section title="The Snapshot Bug and 1D Collapse Retraction" level={1}>
      <Warning>
      <p><strong>Retraction.</strong> V20, V25, and V26 evolution loops reset hidden states to zero BEFORE saving snapshots. All hidden state analysis was on zero vectors. The "1D energy counter" finding was completely wrong.</p>
      </Warning>
      <p>Corrected V22-V24 data: effective rank 5.1-7.3 (not 1-3), energy <M>{"R^2"}</M> = -3.6 to -4.6 (not 1.0), PC1 variance 25-38% (not 95-100%). Hidden states are moderately rich and do NOT encode energy linearly.</p>
      <p><strong>Lesson</strong>: When hidden states appear degenerate, first verify you're analyzing actual post-cycle states, not zeroed reset buffers.</p>
      </Section>

      <Section title="V32: Drought Autopsy" level={1}>
      <p><strong>Status</strong>: Complete. 50 seeds x 30 cycles x 5 droughts each (250 total drought events).</p>
      <p><strong>Question</strong>: What happens DURING drought that determines whether a seed develops high <M>{"\\intinfo"}</M>? V31 showed post-drought bounce predicts final <M>{"\\intinfo"}</M>, but what creates the bounce?</p>
      <p><strong>Result: Integration is trajectory, not event.</strong> Mean <M>{"\\intinfo = 0.086 \\pm 0.032"}</M> (95% CI [0.077, 0.095]). Distribution: 22% HIGH / 46% MODERATE / 32% LOW. Max <M>{"\\intinfo = 0.473"}</M> (seed 23, new all-time record). Mean drought mortality 96.8%.</p>
      <Diagram src="/diagrams/appendix-6.svg" alt="Integration distribution across 50 seeds showing 22% HIGH, 46% MOD, 32% LOW" />
      <p><strong>Key revision from V31</strong>: The first drought bounce does NOT predict final category (<M>{"r = -0.075, p = 0.60"}</M>). What predicts is the <em>mean bounce across all 5 droughts</em> (<M>{"\\rho = 0.599, p = 4.4 \\times 10^{-6}"}</M>). Integration is built by repeatedly bouncing back, not by a single event. <M>{"\\intinfo"}</M> trajectory slope separates categories perfectly (ANOVA <M>{"F = 34.38, p = 6.3 \\times 10^{-10}"}</M>): every HIGH seed has positive slope, every LOW seed has negative slope.</p>
      <p><strong>Robustness is orthogonal to integration</strong> (Mann-Whitney <M>{"p = 0.73"}</M>). Seeds that survive droughts well are not the same seeds that develop high <M>{"\\intinfo"}</M>. Effective rank (mean 8.1) does not differ across categories.</p>
      <Diagram src="/diagrams/appendix-2.svg" alt="The bottleneck furnace: repeated drought-recovery cycles forge integration in HIGH seeds while LOW seeds decline" />
      <Figure src="" alt="Bottleneck furnace animation" caption={<><strong>The Bottleneck Furnace in action.</strong> 256 agents face 5 near-extinction events (93-98% mortality). Survivors rebuild the population each time. The <M>{"\\intinfo"}</M> chart tracks integration climbing — each recovery bounces higher than the last. 12 seconds of simulated evolution compressed from 30 cycles.</>}>
        <video src="/videos/bottleneck-furnace.mp4" autoPlay loop muted playsInline style={{ width: '100%', borderRadius: 8 }} />
      </Figure>
      <Figure src="" alt="Integration trajectory animation" caption={<><strong>Integration Is Biography — Seed 23.</strong> The highest-<M>{"\\intinfo"}</M> seed in 50 (V32) climbs from 0.058 to 0.473 across 30 evolutionary cycles. Each drought (red bands) drops <M>{"\\intinfo"}</M>, but each recovery bounces back higher. The envelope of peaks rises steadily — the furnace forges by compressing, not accumulating.</>}>
        <video src="/videos/integration-trajectory.mp4" autoPlay loop muted playsInline style={{ width: '100%', borderRadius: 8 }} />
      </Figure>
      <CodeFiles files={[
        { name: 'v32_evolution.py', desc: 'Fine-grained drought tracking' },
        { name: 'v32_gpu_run.py', desc: '50-seed GPU runner' },
        { name: 'v32_analysis.py', desc: 'Drought autopsy analysis' },
      ]} />
      </Section>

      <Section title="V33: Contrastive Self-Prediction" level={1}>
      <p><strong>Status</strong>: COMPLETE. <strong>NEGATIVE.</strong></p>
      <p><strong>Hypothesis</strong>: Predicting <M>{"\\Delta_{\\text{actual}} - \\Delta_{\\text{alternative}}"}</M> forces counterfactual representation (rung 8). Standard prediction can be satisfied reactively; contrastive prediction requires representing "what would happen if."</p>
      <p><strong>Result</strong>: Contrastive loss destabilizes gradient learning. Mean <M>{"\\intinfo"}</M> = 0.054 ± 0.015 (late phase), significantly below V27 baseline (0.091). 0% HIGH, 30% MOD, 70% LOW across 10 seeds. Prediction MSE increases 1.5–18.7× over evolution in most seeds — the contrastive signal amplifies after drought cycles, decoupling the gradient from the viability signal. All three pre-registered predictions falsified.</p>
      <CodeFiles files={[
        { name: 'v33_substrate.py', desc: 'Contrastive prediction head' },
        { name: 'v33_evolution.py', desc: 'Evolution loop' },
        { name: 'v33_gpu_run.py', desc: 'GPU runner (10 seeds)' },
      ]} />
      </Section>

      <Section title="V34: Phi-Inclusive Fitness" level={1}>
      <p><strong>Status</strong>: COMPLETE. <strong>MIXED NEGATIVE.</strong></p>
      <p><strong>Hypothesis</strong>: Direct selection for <M>{"\\intinfo"}</M> (fitness = survival_time x (1 + 2<M>{"\\intinfo"}</M>)) pushes HIGH fraction above 30%. Or does it Goodhart?</p>
      <p><strong>Result</strong>: Direct selection for <M>{"\\intinfo"}</M> does not increase the HIGH fraction. 2 HIGH / 3 MOD / 5 LOW across 10 seeds (20% HIGH, within noise of V27's 22%). Late <M>{"\\intinfo"}</M> = 0.079 ± 0.036 (t = -1.08 vs V27, not significant). 2/10 seeds show Goodharting (<M>{"\\intinfo"}</M>-robustness correlation &lt; -0.3). Integration cannot be selected for directly — it must emerge as a byproduct of architectural coupling and trajectory-dependent forging.</p>
      <CodeFiles files={[
        { name: 'v34_evolution.py', desc: 'Phi-inclusive fitness function' },
        { name: 'v34_gpu_run.py', desc: 'GPU runner (10 seeds)' },
      ]} />
      </Section>

      <Section title="V35: Language Emergence" level={1}>
      <p><strong>Status</strong>: Complete (10 seeds).</p>
      <p><strong>Hypothesis</strong>: Language emerges when: (1) partial observability creates information asymmetry (obs_radius=1), (2) discrete channel forces categorical representation (K=8 symbols), (3) cooperative pressure rewards signaling (1.5x bonus for co-consumption), (4) communication range exceeds visual range (comm_radius=5 {"> "} obs_radius=1).</p>
      <p><strong>Result: Referential communication emerges in 10/10 seeds (100%).</strong> Mean symbol entropy 2.48 ± 0.14 bits (83% of maximum, range 2.18–2.67). Resource MI proxy 0.001–0.005 (all positive). All 8 symbols maintained in active use. This <strong>breaks the V20b null</strong> where continuous z-gate signals never departed from 0.5. Discrete symbols under partial observability and cooperative pressure produce referential communication as an <em>inevitability</em>, not a rarity.</p>
      <p><strong>But communication does NOT lift integration.</strong> Mean comm ablation <M>{"\\intinfo"}</M> lift ≈ 0. Late <M>{"\\intinfo = 0.074 \\pm 0.013"}</M> — <em>below</em> V27 baseline (0.090, <M>{"t = -1.78"}</M>). Distribution: 0 HIGH / 7 MOD / 3 LOW. <M>{"\\intinfo"}</M>-MI correlation <M>{"\\rho = 0.07"}</M> (null): language and integration are orthogonal. Communication neither helps nor hurts — it operates on a different axis entirely.</p>
      <p><strong>Language is cheap.</strong> Like affect geometry, referential communication emerges under minimal conditions — partial observability plus cooperative pressure. It sits at rung 4–5 of the emergence ladder. Language does not create dynamics any more than geometry does. The expensive transition remains at rung 8, requiring embodied agency and gradient coupling. Adding communication channels does not help cross it.</p>
      <CodeFiles files={[
        { name: 'v35_substrate.py', desc: 'Discrete communication + cooperative dynamics' },
        { name: 'v35_evolution.py', desc: 'Evolution loop with communication metrics' },
        { name: 'v35_gpu_run.py', desc: 'GPU runner (10 seeds)' },
      ]} />
      </Section>

      <Section title="Integration Landscape: V22–V35" level={1}>
      <Figure src="/images/phi_landscape_v22_v35.png" alt="Integration (Phi) across all experiment conditions V22-V35" caption={<><strong>Integration landscape across all conditions.</strong> Top: mean late-phase Φ with error bars. Green = MLP gradient coupling (baseline), blue = prediction variants, gray = V28 architecture controls, red = negative results, orange = mixed, purple = language. V27 baseline (dashed, Φ = 0.091) is the reference. V29/V31 (social prediction, 10 seeds) is the only condition with mean above baseline, but not significantly so (p = 0.93). V33 (contrastive, 0.054) is significantly below. V35 (language, 0.074) adds communication without lifting integration. Bottom: seed outcome distribution. The 30% HIGH / 30% MOD / 40% LOW split is remarkably stable across conditions — only V33 (70% LOW) and V34 (50% LOW, Goodharting) deviate. Architecture matters; targets and channels do not.</>} />
      </Section>

      <Section title="VLM Convergence Experiment" level={1}>
      <Diagram src="/diagrams/appendix-8.svg" alt="VLM convergence: vision-language models trained on human data recognize affect in protocells" />
      <p><strong>Status</strong>: Complete. Both models tested.</p>
      <p><strong>Core question</strong>: If affect geometry is universal, do systems trained on human affect data (GPT-4o, Claude) independently recognize the same affect signatures in completely uncontaminated substrates?</p>
      <p><strong>Method</strong>: 48 behavioral vignettes extracted from V27/V31 protocell data across 6 conditions (normal foraging, pre-drought abundance, drought onset, drought survival, post-drought recovery, late-stage evolution). Presented to VLMs with purely behavioral descriptions — no affect language, no framework terms, explicitly labeled as artificial systems. Framework predictions computed independently. Convergence measured via Representational Similarity Analysis (RSA) between framework-predicted and VLM-labeled affect spaces.</p>
      <p><strong>Result: STRONG CONVERGENCE.</strong> GPT-4o: RSA <M>{"\\rho = 0.72"}</M> (<M>{"p < 0.0001"}</M>). Claude Sonnet: <M>{"\\rho = 0.54"}</M> (<M>{"p < 0.0001"}</M>). All four pre-registered predictions pass on both models:</p>
      <ul>
      <li><strong>P1</strong>: VLMs label drought onset as fear/anxiety — <strong>PASS</strong> (both: desperation, anxiety, urgency, 8/8 unanimous)</li>
      <li><strong>P2</strong>: VLMs label post-drought recovery as relief/hope — <strong>PASS</strong> (both: relief, cautious optimism)</li>
      <li><strong>P3</strong>: VLMs distinguish HIGH vs LOW late-stage — see condition summary</li>
      <li><strong>P4</strong>: RSA between framework and VLM affect spaces {">"} 0.3 — <strong>PASS</strong> (0.72 and 0.54)</li>
      </ul>
      <p><strong>Robustness check: raw numbers only.</strong> Re-ran with purely numerical descriptions (no narrative framing — just measured quantities like <code>removal_fraction: 0.9800</code>). Convergence <em>increases</em>: GPT-4o <M>{"\\rho = 0.78"}</M>, Claude <M>{"\\rho = 0.72"}</M>. This rules out narrative pattern-matching. The VLMs recognize geometric structure from raw numerical patterns — population dynamics and state update rates are sufficient.</p>
      <p><strong>Theoretical significance</strong>: Two VLMs, trained independently on human data, with no exposure to our framework, produce affect labels that match framework geometric predictions for a system that has never encountered human affect concepts. The convergence happens because both are tapping the same underlying structure: affect geometry arises from the physics of viable self-maintenance, and human language about emotions encodes the same geometry the protocells produce.</p>
      <CodeFiles files={[
        { name: 'vlm_convergence.py', desc: 'Full pipeline: vignette extraction, VLM prompting, RSA analysis' },
        { name: 'vlm_convergence_design.md', desc: 'Pre-registered experiment design' },
      ]} />
      </Section>

      <Section title="Falsification Map" level={1}>
      <table>
      <thead><tr><th>Experiment</th><th>Prediction</th><th>Outcome</th></tr></thead>
      <tbody>
      <tr><td>V10 (MARL)</td><td>Forcing functions create geometry</td><td><strong>Contradicted.</strong> All conditions show alignment; removal increases it.</td></tr>
      <tr><td>Exp 2 (World Model)</td><td><M>{"\\mathcal{C}_{\\text{wm}}"}</M> increases with evolution</td><td>Partial. 100x at bottleneck, flat in general population.</td></tr>
      <tr><td>Exp 3 (Representation)</td><td>Compression and modeling co-emerge</td><td>Partial. Co-emerge under bottleneck only. Compression is cheap.</td></tr>
      <tr><td>Exp 4 (Language)</td><td>Compositional communication</td><td>Not confirmed. Chemical commons but <M>{"\\rho_{\\text{topo}} \\approx 0"}</M>.</td></tr>
      <tr><td>Exp 5 (Counterfactual)</td><td>Reactive-to-detached transition</td><td>Null. Wall at <M>{"\\rho_{\\text{sync}} \\approx 0"}</M>.</td></tr>
      <tr><td>Exp 6 (Self-Model)</td><td>SM emergence with <M>{"\\intinfo"}</M> jump</td><td>Weak. n=1 event at bottleneck.</td></tr>
      <tr><td>Exp 7 (Affect Geometry)</td><td>Tripartite alignment</td><td>Partial. A-C develops over evolution (0.01 to 0.38). A-B null.</td></tr>
      <tr><td>Exp 8 (<M>{"\\iota"}</M>)</td><td>Participatory default, animism</td><td><strong>Confirmed.</strong> <M>{"\\iota \\approx 0.30"}</M>, animism {"> "}1.0 in all 20 snapshots.</td></tr>
      <tr><td>Exp 9 (Normativity)</td><td>Exploitation penalty</td><td>Null. Requires agency.</td></tr>
      <tr><td>Exp 10 (Superorganism)</td><td><M>{"\\intinfo_G > \\sum \\intinfo_i"}</M></td><td>Not confirmed. Ratio 1-12%, increasing.</td></tr>
      <tr><td>Exp 11 (Entanglement)</td><td>Co-emergence clusters</td><td>Not confirmed. Different cluster structure.</td></tr>
      <tr><td>Exp 12 (Capstone)</td><td>Seven criteria for identity thesis</td><td>All met (moderate/weak). Geometry confirmed.</td></tr>
      <tr><td>V19 (Furnace)</td><td>Selection vs creation</td><td><strong>Creation confirmed</strong> 2/3 seeds.</td></tr>
      <tr><td>V20 (<M>{"\\rho"}</M> wall)</td><td><M>{"\\rho_{\\text{sync}} > 0.1"}</M></td><td><strong>Confirmed.</strong> 0.21 from cycle 0.</td></tr>
      <tr><td>V22-V24 (Prediction)</td><td>Prediction {"\u2192"} integration</td><td>Not confirmed. Linear readout always decomposable.</td></tr>
      <tr><td>V27 (MLP)</td><td>Nonlinear head {"\u2192"} <M>{"\\intinfo \\uparrow"}</M></td><td><strong>Confirmed</strong> (seed 7: 0.245). Seed-dependent.</td></tr>
      <tr><td>V28 (Width)</td><td>Bottleneck width matters</td><td>Not confirmed. Mechanism is gradient coupling.</td></tr>
      <tr><td>V29/V31 (Social)</td><td>Social target lifts <M>{"\\intinfo"}</M></td><td>Not confirmed. 3-seed fluke; 10-seed: <M>{"p = 0.93"}</M>.</td></tr>
      <tr><td>V30 (Dual)</td><td>Self+social {"> "} either</td><td>Negative. Gradient imbalance; self colonizes.</td></tr>
      <tr><td>V31 (Seeds)</td><td>Seed distribution</td><td><strong>Confirmed:</strong> 30/30/40 split. Post-drought bounce <M>{"r = 0.997"}</M>.</td></tr>
      <tr><td>V32 (Autopsy)</td><td>First bounce predicts category</td><td><strong>Revised:</strong> First bounce NOT predictive (<M>{"p = 0.60"}</M>). Mean bounce across all droughts IS (<M>{"\\rho = 0.60, p < 10^{-5}"}</M>). Trajectory, not event.</td></tr>
      <tr><td>V35 (Language)</td><td>Referential communication emerges</td><td><strong>Confirmed:</strong> 10/10 seeds (100%). But does NOT lift <M>{"\\intinfo"}</M>. Language is cheap.</td></tr>
      <tr><td>VLM Conv.</td><td>VLMs recognize affect in protocells (RSA {"> "} 0.3)</td><td><strong>Confirmed:</strong> GPT-4o <M>{"\\rho = 0.72"}</M>, Claude <M>{"\\rho = 0.54"}</M>. Raw numbers: 0.78, 0.72.</td></tr>
      </tbody>
      </table>
      <Figure src="" alt="Falsification scoreboard animation" caption={<><strong>Falsification Scoreboard.</strong> 7 confirmed, 7 contradicted, 1 revised. The framework survives not by being right everywhere, but by being wrong in specific, informative ways. Each contradiction sharpened the theory — the forcing function failure led to the geometry/dynamics distinction; the social prediction failure revealed the gradient interference pattern; the language failure established the rung 4-5 / rung 8 boundary.</>}>
        <video src="/videos/falsification-scoreboard.mp4" autoPlay loop muted playsInline style={{ width: '100%', borderRadius: 8 }} />
      </Figure>
      </Section>

      <Section title="Summary" level={1}>
      <Diagram src="/diagrams/appendix-0.svg" alt="The emergence ladder: 10 rungs from affect dimensions to normativity, with experimental status for each" />
      <p>Thirty-five experiment versions across four substrates and fifty seeds at scale. Twelve numbered measurement experiments. One VLM convergence study. Five current priorities. The program has established:</p>
      <ol>
      <li><strong>Geometry is cheap, dynamics are expensive.</strong> Affect geometry arises from the minimal conditions of multi-agent survival (V10, Exp 7-8, Exp 12). Affect dynamics require embodied agency (V20), graduated stress exposure (V19), and non-decomposable prediction architecture (V27).</li>
      <li><strong>Two architectural walls.</strong> The sensory-motor wall (<M>{"\\rho_{\\text{sync}} \\approx 0"}</M>) is broken by genuine action-observation loops (V20). The decomposability wall is broken by 2-layer gradient coupling (V27). Both are necessary.</li>
      <li><strong>Integration is stochastic.</strong> ~30% of seeds develop high <M>{"\\intinfo"}</M> regardless of architecture or prediction target (V31). The predictor is post-drought recovery dynamics (<M>{"r = 0.997"}</M>), not initial conditions. Integration is biographical.</li>
      <li><strong>The bottleneck furnace is generative.</strong> Near-extinction forges integration capacity (V19). Repeated drought recovery is the mechanism (V31, V32). The furnace does not select for pre-existing integration — it creates it. V32 (50 seeds) reveals that integration is trajectory, not event: the mean bounce across all 5 droughts predicts final category (<M>{"\\rho = 0.599, p < 10^{-5}"}</M>), but the first bounce alone does not (<M>{"p = 0.60"}</M>). Integration is built by the sustained pattern of recovery, not by a single crisis.</li>
      <li><strong>Prediction target doesn't matter.</strong> Self vs social prediction produces the same <M>{"\\intinfo"}</M> distribution (V31, <M>{"p = 0.93"}</M>). What matters is the gradient architecture (linear vs 2-layer) and the evolutionary trajectory.</li>
      <li><strong>Language is cheap.</strong> Referential communication emerges in 100% of seeds under partial observability with cooperative pressure (V35). But it does not lift integration — Φ-MI correlation is null (<M>{"\\rho = 0.07"}</M>), meaning language and integration operate on orthogonal axes. Like geometry, language is an inevitability of survival under information asymmetry. Like geometry, it does not cross the rung 8 wall.</li>
      <li><strong>The geometry is universal.</strong> VLMs trained on human data — with no exposure to the framework — independently recognize the same affect signatures in completely uncontaminated protocell systems (RSA <M>{"\\rho = 0.54"}</M>–<M>{"0.78"}</M>, <M>{"p < 0.0001"}</M>). The convergence holds and <em>strengthens</em> when narrative framing is removed and only raw numbers remain. Affect geometry arises from the structure of viable self-maintenance, not from biological contingency.</li>
      </ol>
      <p>The framework is not confirmed. It is informed. What it predicted about geometry was too weak — geometry is cheaper than expected, and now independently validated by cross-substrate convergence. What it predicted about dynamics was too strong — dynamics require specific architectural affordances the theory didn't anticipate. The interesting question is no longer "does the geometry exist?" (it does, trivially, and VLMs trained on human data agree) but "what determines which systems develop the dynamics that make the geometry experientially real?"</p>
      </Section>
    </>
  );
}
