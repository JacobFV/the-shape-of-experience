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
      <p>Every claim in this book is either tested, testable, or honestly labeled as speculative. This appendix catalogs the full experimental program: what has been run, what the results show, and what remains. It is a living document — updated as experiments complete.</p>
      </Logos>
      <Section title="Completed Experiments" level={1}>
      <Section title="V10: MARL Forcing Function Ablation" level={2}>
      <p><strong>Question</strong>: Do forcing functions create geometric affect alignment?</p>
      <p><strong>Result</strong>: No. All 7 conditions show significant alignment (<M>{"\\rho > 0.21"}</M>, <M>{"p < 0.0001"}</M>). Removing forcing functions slightly <em>increases</em> alignment. Affect geometry is a baseline property of multi-agent survival.</p>
      <p><strong>Implication</strong>: Geometry is cheap. The structure arises from the minimal conditions of survival under uncertainty.</p>
      </Section>
      <Section title="V11.0–V11.7: Lenia CA Evolution Series" level={2}>
      <p><strong>Question</strong>: Can evolution under survival pressure produce biological-like integration dynamics?</p>
      <p><strong>Key findings</strong>:</p>
      <ul>
      <li>Yerkes-Dodson is universal: mild stress increases <M>{"\\intinfo"}</M> by 60–200%</li>
      <li>Curriculum training is the only intervention that improves novel-stress generalization (+1.2 to +2.7pp)</li>
      <li>The locality ceiling: convolutional physics cannot produce <M>{"\\intinfo"}</M> increase under severe threat</li>
      </ul>
      <p><strong>Implication</strong>: Training regime matters more than substrate complexity. And something beyond convolution is needed.</p>
      </Section>
      <Section title="V12: Attention-Based Lenia" level={2}>
      <p><strong>Question</strong>: Does state-dependent interaction topology enable the biological integration pattern?</p>
      <p><strong>Result</strong>: Partially. Evolvable attention shows <M>{"\\intinfo"}</M> increase in 42% of cycles (vs 3% for convolution). +2.0pp shift — largest single-intervention effect. But robustness stabilizes near 1.0 without further improvement.</p>
      <p><strong>Implication</strong>: Attention is necessary but not sufficient. The system reaches the integration threshold without crossing it. Missing ingredient: individual-level plasticity.</p>
      </Section>
      <Section title="V13: Content-Based Coupling Lenia" level={2}>
      <p><strong>Question</strong>: Does content-based interaction topology (simpler than learned attention) produce integration under stress?</p>
      <p><strong>Result</strong>: Yes, intermittently. Three seeds, 30 cycles each (<M>{"C{=}16"}</M>, <M>{"N{=}128"}</M>). Mean robustness <M>{"0.923"}</M>, peak <M>{"1.052"}</M> at population bottlenecks. 30% of patterns show <M>{"\\intinfo"}</M> increase under stress. Robustness {">"} 1.0 only appears when population drops below ~50 — bottleneck events select for integration. Two evolutionary strategies: open coupling (<M>{"\\tau \\to 0"}</M>, large stable populations) vs selective coupling (<M>{"\\tau \\to 0.86"}</M>, volatile populations with occasional robustness {">"} 1.0).</p>
      <p><strong>Implication</strong>: Content-dependent topology enables the biological pattern under selection pressure. The mechanism may be closer to symbiogenesis (composition of functional units) than classical Darwinian optimization. Mutual legibility between patterns — enabled by content coupling — allows compositional encounter. Experiment 0 substrate is viable.</p>
      </Section>
      </Section>

      <Section title="The Emergence Experiment Program" level={1}>
      <p>The next phase tests whether the capacities the book describes — world modeling, abstraction, communication, counterfactual reasoning, self-modeling — <em>co-emerge</em> in a single substrate with zero human contamination. Rather than treating these as separate experiments on separate substrates, we define a single evolving measurement framework that tracks multiple quantities simultaneously, looking for correlated transitions.</p>

      <Section title="Experiment 0: Substrate Engineering (COMPLETED)" level={2}>
      <p>The foundation. Lenia with <strong>state-dependent interaction topology</strong> — cells selectively couple with distant cells based on state similarity:</p>
      <Eq>{"K_i(j) = K_{\\text{base}}(|i-j|) \\cdot \\sigma\\!\\bigl(\\langle h(s_i),\\, h(s_j) \\rangle - \\tau\\bigr)"}</Eq>
      <p>Plus lethal resource dynamics ({">"} 50% naive mortality during drought). Implemented as V13. The substrate sustains 50–180 patterns across 30 evolution cycles, with content-similarity modulation enabling state-dependent interaction topology at 568 steps/s (CPU, <M>{"C{=}8"}</M>). Confirmed: lethal drought (82% mortality at <M>{"C{=}8"}</M>), curriculum stress schedule, and population rescue mechanism for robustness. <M>{"\\tau"}</M> and gate steepness <M>{"\\beta"}</M> are evolvable. Ready for Experiments 1–7.</p>
      </Section>

      <Section title="Experiment 1: Emergent Existence (COMPLETED)" level={2}>
      <p>Rungs 1–3. Patterns persist, maintain boundaries, respond to perturbation. Established by V11–V12. The operational definition: a pattern exists when its internal correlations exceed background and it maintains identity over time.</p>
      </Section>

      <Section title="Experiment 2: Emergent World Model" level={2}>
      <p>When does a pattern's internal state carry predictive information about the environment beyond what's available from current observations?</p>
      <p><strong>Key measure</strong>: The <em>prediction gap</em> — train two predictors, one with access to the pattern's internals and one without. The gap <M>{"\\mathcal{W}(t, \\tau) = \\mathcal{L}[f_{\\text{env}}] - \\mathcal{L}[f_{\\text{full}}]"}</M> measures how much the pattern "knows" about the future that isn't readable from the present environment.</p>
      <p><strong>Derived</strong>: World model horizon <M>{"H_{\\text{wm}}"}</M> (maximum useful prediction distance) and world model capacity <M>{"\\mathcal{C}_{\\text{wm}}"}</M> (total predictive information across horizons).</p>
      <p><strong>Prediction</strong>: <M>{"\\mathcal{C}_{\\text{wm}}"}</M> increases with evolutionary generation in attention substrates but remains near zero in convolutional substrates (locality ceiling).</p>
      </Section>

      <Section title="Experiment 3: Internal Representation Structure" level={2}>
      <p>When do patterns develop <strong>low-dimensional, compositional</strong> representations? Measured by effective dimensionality <M>{"d_{\\text{eff}}"}</M> (compression), disentanglement score <M>{"\\mathcal{D}"}</M> (each latent dimension tracks one environmental feature), and abstraction level <M>{"\\mathcal{A} = 1 - d_{\\text{eff}} / \\min(|B|, M)"}</M>.</p>
      <p><strong>Key prediction</strong>: <M>{"d_{\\text{eff}}"}</M> should track <M>{"\\mathcal{C}_{\\text{wm}}"}</M> — compression and modeling co-emerge.</p>
      </Section>

      <Section title="Experiment 4: Emergent Language" level={2}>
      <p>When do patterns develop structured, compositional communication? Measure signal entropy (below noise = structured), contingency on internal state (not just reflexive), compositionality via topographic similarity <M>{"\\rho_{\\text{topo}}"}</M> (similar contexts produce similar signals).</p>
      <p><strong>Prediction</strong>: Language emerges <em>after</em> world models and compression — you need something to communicate about, the capacity to compress it, and multi-agent coordination pressure.</p>
      </Section>

      <Section title="Experiment 5: Counterfactual Detachment" level={2}>
      <p>When do patterns decouple from external driving and run offline world model rollouts? Measure external synchrony <M>{"\\rho_{\\text{sync}}"}</M> (correlation between internal updates and boundary input). During detachment events (<M>{"\\rho_{\\text{sync}}"}</M> below threshold), measure whether the offline trajectory is more predictive of future environment than a reactive-mode trajectory of equal duration.</p>
      <p>Positive counterfactual simulation score = the pattern imagines usefully. This is the counterfactual weight dimension measured in the substrate.</p>
      </Section>

      <Section title="Experiment 6: Self-Model Emergence" level={2}>
      <p>When does a pattern predict <em>itself</em> better than an external observer can? The self-effect ratio <M>{"\\rho"}</M> measures how much the pattern's own actions dominate its observation stream. When <M>{"\\rho > 0.5"}</M>, self-modeling becomes the cheapest path to prediction.</p>
      <p><strong>Key prediction from the thesis</strong>: Self-model emergence should correlate with a jump in <M>{"\\intinfo"}</M>, because self-referential loops couple all internal components.</p>
      </Section>

      <Section title="Experiment 7: Affect Geometry Verification" level={2}>
      <p>The <strong>tripartite alignment test</strong>. For patterns with world models and self-models: extract structural affect dimensions from internals (Space A), from emitted signals (Space B), and from behavior (Space C). Test alignment via RSA across all three spaces. Then perturb bidirectionally: inject false signals, modify coupling parameters, change environment. If perturbations propagate through all three spaces, the structural identity is supported.</p>
      </Section>

      <Section title="Experiments 8–10: Social and Normative Emergence" level={2}>
      <p>Whether on Lenia or another substrate — these test the book's social-scale claims:</p>
      <ul>
      <li><strong>Experiment 8</strong> (<M>{"\\iota"}</M> emergence): Do patterns develop modulable perceptual coupling — switching between modeling others as agents vs. objects? The thesis predicts participatory perception (low <M>{"\\iota"}</M>) as default.</li>
      <li><strong>Experiment 9</strong> (Proto-normativity): Does the viability gradient penalize exploitation even when exploitation is locally rewarding? Measure valence, self-model salience, and integration during cooperative vs. exploitative behavior.</li>
      <li><strong>Experiment 10</strong> (Superorganism integration): Does collective <M>{"\\intinfo_G > \\sum_i \\intinfo_i"}</M>? If so, information exists at the group level that doesn't exist in any individual — the "gods" framework has empirical support.</li>
      </ul>
      </Section>

      <Section title="Experiment 11: Entanglement Analysis" level={2}>
      <p>Track all quantities simultaneously across evolutionary generations. Compute the correlation matrix. Test:</p>
      <ul>
      <li><strong>Co-emergence</strong>: <M>{"\\mathcal{C}_{\\text{wm}}"}</M>, <M>{"\\mathcal{A}"}</M>, and <M>{"\\mathcal{I}_{\\text{img}}"}</M> should be strongly correlated wherever any is nonzero</li>
      <li><strong>Partial separability</strong>: Language (<M>{"\\rho_{\\text{topo}}"}</M>) should lag, requiring multi-agent coordination as additional forcing</li>
      <li><strong>Threshold structure</strong>: Punctuated equilibria — generations where several quantities spike together</li>
      </ul>
      </Section>

      <Section title="Experiment 12: Identity Thesis Capstone" level={2}>
      <p>The tripartite alignment test (Experiment 7) run on the most complex patterns that emerge from the full program. This is the test: if we build a system that models the world, models itself, communicates with others, imagines futures, and shows structured internal states tracking its viability gradient — and the structural geometry aligns with what we observe in biological systems — what reason remains to deny that the system has affect?</p>
      <p>If the answer is "none," the identity thesis is supported. If the answer is "because it's just physics" — that is the point.</p>
      </Section>
      </Section>

      <Section title="Falsification Map" level={1}>
      <table>
      <thead><tr><th>Experiment</th><th>If we find...</th><th>Then...</th></tr></thead>
      <tbody>
      <tr><td>2</td><td>No world models in attention substrate</td><td>State-dependent coupling insufficient. Need richer physics.</td></tr>
      <tr><td>3</td><td>No compression co-emerging with modeling</td><td>Compression and prediction are separable. Entanglement prediction fails.</td></tr>
      <tr><td>4</td><td>No structured communication under coordination pressure</td><td>Language requires something beyond survival pressure.</td></tr>
      <tr><td>5</td><td>No useful offline processing</td><td>Counterfactual reasoning requires more than world models.</td></tr>
      <tr><td>6</td><td>No self-model emergence or no <M>{"\\intinfo"}</M> jump</td><td>Self-modeling and integration are not coupled. Core thesis weakens.</td></tr>
      <tr><td>7</td><td>Tripartite alignment fails</td><td>Affect geometry is a measurement artifact, not a real structural property.</td></tr>
      <tr><td>9</td><td>No valence asymmetry under exploitation</td><td>Normativity is not structural. Is-ought dissolution fails.</td></tr>
      <tr><td>10</td><td><M>{"\\intinfo_G = \\sum \\intinfo_i"}</M></td><td>Superorganism integration is additive. "Gods" framework = metaphor.</td></tr>
      <tr><td>12</td><td>No alignment in fully uncontaminated substrate</td><td>Affect geometry requires human-like training. Universality claim fails.</td></tr>
      </tbody>
      </table>
      </Section>

      <Section title="Summary" level={1}>
      <p>Three phases:</p>
      <ol>
      <li><strong>What has been tested</strong>: Affect geometry is cheap (V10). Dynamics require composition under selection pressure (V11–V13). Attention/content coupling is necessary; population bottleneck events produce the strongest integration signals.</li>
      <li><strong>What is ready to test</strong>: The emergence program (Experiments 1–7). The V13 substrate is operational. Can world models, abstraction, language, counterfactual reasoning, and self-modeling co-emerge in an uncontaminated substrate?</li>
      <li><strong>What comes after</strong>: Social-scale claims (Experiments 8–10), entanglement analysis (11), and the identity thesis capstone (12).</li>
      </ol>
      <p>The theory is falsifiable. The experiments are specified. The substrate is built. The question is whether the predictions hold when we remove every trace of human contamination and let the physics speak for itself.</p>
      </Section>
    </>
  );
}
