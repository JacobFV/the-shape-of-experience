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
      <p>Every claim in this book is either tested, testable, or honestly labeled as speculative. This appendix catalogs the full experimental program: what has been run, what the results show, and what remains. It is a living document — updated as experiments are completed.</p>
      </Logos>
      <Section title="Completed Experiments" level={1}>
      <Section title="V2–V9: LLM Affect Signatures" level={2}>
      <p><strong>Question</strong>: Is the geometric affect framework measurable in artificial systems?</p>
      <p><strong>Method</strong>: Progressive refinement across 8 versions. LLM and RL agents in survival environments. Affect dimensions extracted from internal representations.</p>
      <p><strong>Key findings</strong>:</p>
      <ul>
      <li>The affect space is coherent and measurable in LLM agents</li>
      <li>Processing valence <M>{"\\neq"}</M> content valence (critical distinction)</li>
      <li>LLM agents show <em>opposite dynamics</em> to biological systems: <M>{"\\intinfo \\downarrow"}</M>, <M>{"\\mathcal{SM} \\downarrow"}</M>, <M>{"\\arousal \\downarrow"}</M> under threat</li>
      </ul>
      <p><strong>Limitation</strong>: All LLM agents are trained on human language. The affect structure may be inherited from training data rather than emerging from the agents' own dynamics. This is the contamination problem that V13 addresses.</p>
      </Section>
      <Section title="V10: MARL Forcing Function Ablation" level={2}>
      <p><strong>Question</strong>: Do forcing functions (partial observability, long horizons, world models, self-prediction, intrinsic motivation, credit assignment) create geometric affect alignment?</p>
      <p><strong>Method</strong>: 7 conditions (full model + 6 single ablations) <M>{"\\times"}</M> 3 seeds <M>{"\\times"}</M> 200k steps. RSA with Mantel test.</p>
      <p><strong>Result</strong>: All conditions show highly significant alignment (<M>{"\\rho > 0.21"}</M>, <M>{"p < 0.0001"}</M>). Removing forcing functions slightly <em>increases</em> alignment.</p>
      <table>
      <thead><tr><th>Condition</th><th>RSA <M>{"\\rho"}</M></th><th><M>{"\\pm"}</M> std</th></tr></thead>
      <tbody>
      <tr><td>full</td><td>0.212</td><td>0.058</td></tr>
      <tr><td>no_partial_obs</td><td>0.217</td><td>0.016</td></tr>
      <tr><td>no_long_horizon</td><td>0.215</td><td>0.027</td></tr>
      <tr><td>no_world_model</td><td>0.227</td><td>0.005</td></tr>
      <tr><td>no_self_prediction</td><td>0.240</td><td>0.022</td></tr>
      <tr><td>no_intrinsic_motivation</td><td>0.212</td><td>0.011</td></tr>
      <tr><td>no_delayed_rewards</td><td>0.254</td><td>0.051</td></tr>
      </tbody>
      </table>
      <p><strong>Implication</strong>: Affect geometry is a baseline property of multi-agent survival. The geometry is cheaper than predicted. Forcing functions add agent capabilities, not affect structure.</p>
      <p><strong>Limitation</strong>: All V10 agents are multi-agent. The geometry may require social interaction rather than being universal. V10 agents also use pretrained components — language contamination not fully controlled.</p>
      </Section>
      <Section title="V11.0–V11.7: Lenia CA Evolution Series" level={2}>
      <p><strong>Question</strong>: Can evolution under survival pressure produce biological-like integration dynamics — specifically, <M>{"\\intinfo"}</M> <em>increasing</em> under threat?</p>
      <p><strong>Method</strong>: Lenia (continuous cellular automata) with resource dynamics. 8 substrate conditions across hundreds of GPU-hours.</p>
      <table>
      <thead><tr><th>Version</th><th>Substrate</th><th><M>{"\\Delta\\intinfo"}</M> (drought)</th><th>Key lesson</th></tr></thead>
      <tbody>
      <tr><td>V11.0</td><td>Naive Lenia</td><td><M>{"-6.2\\%"}</M></td><td>Decomposition baseline</td></tr>
      <tr><td>V11.1</td><td>Homogeneous evolution</td><td><M>{"-6.0\\%"}</M></td><td>Selection without variation cannot innovate</td></tr>
      <tr><td>V11.2</td><td>Heterogeneous chemistry</td><td><M>{"-3.8\\%"}</M> vs naive <M>{"-5.9\\%"}</M></td><td>+2.1pp shift toward biological pattern</td></tr>
      <tr><td>V11.5</td><td>Hierarchical 4-tier</td><td><M>{"-9.3\\%"}</M> (evolved)</td><td>Stress overfitting: high <M>{"\\intinfo"}</M> = fragile <M>{"\\intinfo"}</M></td></tr>
      <tr><td>V11.7</td><td>Curriculum training</td><td>+1.2 to +2.7pp generalization</td><td>Training regime &gt; substrate complexity</td></tr>
      </tbody>
      </table>
      <p><strong>Synthesis</strong>:</p>
      <ol>
      <li><strong>Yerkes-Dodson is universal</strong>: Mild stress increases <M>{"\\intinfo"}</M> by 60–200% across all conditions</li>
      <li><strong>Evolution produces fragile integration</strong>: Tight coupling = catastrophic failure under severe stress</li>
      <li><strong>Curriculum training is the only intervention that improves novel-stress generalization</strong></li>
      <li><strong>The locality ceiling</strong>: Convolutional physics (fixed interaction topology) cannot produce biological-like <M>{"\\intinfo"}</M> increase under threat</li>
      </ol>
      </Section>
      <Section title="V12: Attention-Based Lenia" level={2}>
      <p><strong>Question</strong>: Does state-dependent interaction topology (attention) enable the biological integration pattern that local physics cannot produce?</p>
      <p><strong>Method</strong>: Replace FFT convolution with windowed self-attention. 3 conditions <M>{"\\times"}</M> 3 seeds.</p>
      <table>
      <thead><tr><th>Condition</th><th>Robustness</th><th>% cycles <M>{"\\intinfo \\uparrow"}</M></th><th>Notes</th></tr></thead>
      <tbody>
      <tr><td>A: Fixed-local attention</td><td>N/A (extinct)</td><td>0%</td><td>30+ consecutive extinctions</td></tr>
      <tr><td>B: Evolvable attention</td><td>1.001</td><td>42%</td><td>+2.0pp over convolution</td></tr>
      <tr><td>C: Convolution baseline</td><td>0.981</td><td>3%</td><td>Life without integration</td></tr>
      </tbody>
      </table>
      <p><strong>Interpretation</strong>: Attention is necessary but not sufficient. The system reaches the integration threshold without clearly crossing it. The ordering is: convolution sustains life without integration; fixed attention cannot sustain life; evolvable attention sustains life <em>with</em> integration at the threshold. What remains missing is individual-level plasticity — within-lifetime adaptation rather than population-level selection.</p>
      </Section>
      </Section>
      <Section title="Next-Phase Experiments" level={1}>
      <p>The following experiments are designed to test the framework's deepest claims. They are ordered by theoretical importance.</p>
      <Experiment title="V13: Uncontaminated Language Emergence">
      <p><strong>Question</strong>: Does affect structure emerge in communicating agents with no exposure to human affect concepts?</p>
      <p><strong>Design</strong>: Multi-agent RL (4–8 agents) with randomly-initialized transformers (no pretraining). Survival environment with seasonal resources, predators, weather. Communication via learnable discrete tokens (vocabulary 32–128). No human language, no pretrained embeddings, no affect labels.</p>
      <p><strong>Measurements</strong>:</p>
      <ul>
      <li>RSA between information-theoretic and embedding-predicted affect spaces (as in V10)</li>
      <li><M>{"\\text{MI}(\\text{sender\\_affect}; \\text{message})"}</M> — do messages encode affective state?</li>
      <li><M>{"\\text{MI}(\\text{message}; \\Delta\\text{receiver\\_affect})"}</M> — do messages modulate recipient affect?</li>
      </ul>
      <p><strong>Predictions</strong>: Geometric alignment should be significant. Communication should carry affect-relevant information. Warning signals should emerge with high arousal + negative valence content.</p>
      <p><strong>Failure mode</strong>: RSA <M>{"\\rho \\approx 0"}</M> without human language contamination. Would indicate that affect geometry requires human-like training data, not just survival pressure. Universality claim fails.</p>
      <p><strong>Status</strong>: Not yet implemented. Extends V10 infrastructure.</p>
      </Experiment>
      <Experiment title="V14: Solitary Rumination and World Model Detachment">
      <p><strong>Question</strong>: Do internal dynamics alone produce structured affect? Can a system have affect without external input?</p>
      <p><strong>Design</strong>: Single agent with explicit world model, trained on rich environment. Three phases: (1) normal training with sensory input; (2) sensory disconnection — world model runs on its own predictions ("imagination mode"); (3) reconnection.</p>
      <p><strong>Predictions</strong>:</p>
      <ol>
      <li><M>{"\\mathcal{CF}"}</M> increases during disconnection (more resources to non-actual trajectories)</li>
      <li><M>{"\\intinfo"}</M> initially increases then slowly decreases (world model collapses to attractors)</li>
      <li><M>{"\\effrank"}</M> decreases (fewer degrees of freedom than reality)</li>
      <li><M>{"\\valence"}</M> drifts negative (viability estimates degrade without calibration)</li>
      <li>Reconnection: <M>{"\\arousal"}</M> spike, then normalization</li>
      <li><M>{"\\mathcal{SM}"}</M> increases during disconnection (self is most salient signal)</li>
      </ol>
      <p><strong>Why this matters</strong>: Computational equivalent of meditation (voluntary detachment), dreaming (involuntary), dissociation (pathological), and solitary confinement (forced). If predictions hold, the framework explains why isolation is psychologically devastating, why meditation requires training, and why reconnection is disorienting.</p>
      <p><strong>Failure mode</strong>: No structured affect during offline mode. Would indicate affect requires external input — the existential burden claim weakens.</p>
      <p><strong>Status</strong>: Not yet implemented. Requires new single-agent architecture.</p>
      </Experiment>
      <Experiment title="V15: World Model Derailment">
      <p><strong>Question</strong>: Does systematic prediction error have a specific affect signature?</p>
      <p><strong>Design</strong>: Agent trained on environment A, transferred to environment B (different dynamics). Four conditions: gradual shift, sudden shift, partial shift, adversarial shift.</p>
      <p><strong>Predictions</strong>:</p>
      <ol>
      <li><M>{"\\arousal"}</M> spikes proportional to prediction error magnitude</li>
      <li><M>{"\\valence"}</M> goes negative (unreliable viability estimates)</li>
      <li><M>{"\\mathcal{SM}"}</M> increases (agent needs to recalibrate self-model)</li>
      <li><M>{"\\effrank"}</M> expands then contracts (more hypotheses, then new model crystallizes)</li>
      <li>Gradual shift: less arousal, longer adaptation (boiling frog)</li>
      <li>Adversarial shift: persistent high arousal + negative valence (anxiety/paranoia analog)</li>
      </ol>
      <p><strong>Why this matters</strong>: Tests the framework's prediction about culture shock, cognitive dissonance, and paradigm shifts. The adversarial condition may produce the affect signature of paranoia — a world model that is <em>actively</em> being contradicted.</p>
      <p><strong>Failure mode</strong>: No consistent affect signature. Arousal is not belief-update rate, or the other dimensions don't respond systematically.</p>
      <p><strong>Status</strong>: Not yet implemented. Modifies V10 environment.</p>
      </Experiment>
      <Experiment title="V16: Affect Contagion in Multi-Agent Communication">
      <p><strong>Question</strong>: Does affect propagate through communication?</p>
      <p><strong>Design</strong>: Extension of V13. One agent faces local threat while others are safe. Measure whether communication from the threatened agent shifts recipients' affect states.</p>
      <p><strong>Key measure</strong>: <M>{"\\text{MI}(\\text{sender\\_affect}; \\Delta\\text{receiver\\_affect} \\mid \\text{receiver\\_local\\_obs})"}</M></p>
      <p><strong>Predictions</strong>: Negative valence propagates faster than positive. Arousal is most contagious. Integration contagion requires shared world-model context.</p>
      <p><strong>Failure mode</strong>: MI <M>{"\\approx 0"}</M>. Affect is private, not communicable. Parts IV–V (social bonds, gods) lose their empirical foundation.</p>
      <p><strong>Status</strong>: Not yet implemented. Extends V13.</p>
      </Experiment>
      <Experiment title="V17: Proto-Normativity Detection">
      <p><strong>Question</strong>: Does the affect system register a difference between cooperative and exploitative action?</p>
      <p><strong>Design</strong>: Multi-agent environment. After cooperation emerges, introduce opportunity for trust exploitation. Measure affect dimensions during cooperative vs exploitative choices.</p>
      <p><strong>Predictions</strong>: Exploitation produces lower valence than cooperation even when more rewarding. <M>{"\\mathcal{SM}"}</M> increases during exploitation. Response time is longer for exploitation. After repeated exploitation, valence penalty diminishes (desensitization).</p>
      <p><strong>Failure mode</strong>: No valence difference. Normativity is not structural — the is-ought dissolution fails.</p>
      <p><strong>Status</strong>: Not yet implemented. Extends V13.</p>
      </Experiment>
      <Experiment title="V18: Superorganism Integration Measurement">
      <p><strong>Question</strong>: Can multi-agent groups develop collective integration that exceeds the sum of individual integrations?</p>
      <p><strong>Design</strong>: 8–16 agents with communication, specialization, resource sharing. Group-level threat requiring coordination.</p>
      <p><strong>Key measure</strong>: <M>{"\\intinfo_G > \\sum_i \\intinfo_i"}</M>? Partition the group into subgroups and measure prediction loss.</p>
      <p><strong>Predictions</strong>: <M>{"\\intinfo_G > 0"}</M> when agents coordinate. <M>{"\\intinfo_G > \\sum_i \\intinfo_i"}</M> only with information sharing. Group-level threat increases <M>{"\\intinfo_G"}</M>.</p>
      <p><strong>Failure mode</strong>: <M>{"\\intinfo_G \\approx \\sum_i \\intinfo_i"}</M>. Superorganism integration is additive. The "gods" framework collapses to metaphor.</p>
      <p><strong>Status</strong>: Not yet implemented.</p>
      </Experiment>
      <Experiment title="V19: Inhibition Coefficient (ι) Operationalization">
      <p><strong>Question</strong>: Is <M>{"\\iota"}</M> a measurable parameter, and is it scalar or vector?</p>
      <p><strong>Design (computational)</strong>: RL agents with self-model module. Vary conditions: social vs mechanical environment, survival pressure vs safety, novel vs familiar stimuli. Measure 4 proxies:</p>
      <ol>
      <li>Agency attribution rate (how often the world model attributes goals to objects)</li>
      <li>Affect-perception coupling (<M>{"\\text{MI}(\\text{perceptual features}; \\text{affect state})"}</M>)</li>
      <li>Integration (<M>{"\\intinfo"}</M>)</li>
      <li>Self-model activation</li>
      </ol>
      <p><strong>Prediction</strong>: All 4 proxies load on a single factor if <M>{"\\iota"}</M> is scalar. Social environments produce lower <M>{"\\iota"}</M>. Survival pressure reduces <M>{"\\iota"}</M>.</p>
      <p><strong>Design (human, when IRB available)</strong>: Heider-Simmel animations (agency attribution), affect-perception coupling (self-report + physiology), Kelemen paradigm (teleological reasoning), mismatch negativity amplitude (EEG).</p>
      <p><strong>Failure mode</strong>: Proxies don't correlate. <M>{"\\iota"}</M> is not a unitary construct. May need vector treatment.</p>
      <p><strong>Status</strong>: Not yet implemented.</p>
      </Experiment>
      </Section>
      <Section title="Human Studies (Require IRB and Funding)" level={1}>
      <p>Eight pre-registered study protocols have been developed. These require institutional review board approval and funding but are ready for implementation.</p>
      <table>
      <thead><tr><th>#</th><th>N</th><th>Method</th><th>Key prediction</th></tr></thead>
      <tbody>
      <tr><td>1</td><td>500</td><td>CFA factor analysis</td><td>Multi-factor model beats 2-factor (valence <M>{"\\times"}</M> arousal)</td></tr>
      <tr><td>2</td><td>100</td><td>Ambulatory + physiology</td><td>Valence correlates with HRV, cortisol, threat proximity</td></tr>
      <tr><td>3</td><td>60</td><td>EEG + meditation</td><td><M>{"\\intinfo"}</M> correlates with experiential unity</td></tr>
      <tr><td>4</td><td>200</td><td>Cultural exposure</td><td>Different art forms produce distinct affect signatures</td></tr>
      <tr><td>5</td><td>80</td><td>Flow induction</td><td>Flow = low <M>{"\\mathcal{SM}"}</M>, high <M>{"\\intinfo"}</M>, positive <M>{"\\valence"}</M></td></tr>
      <tr><td>6</td><td>90</td><td>Meditation training</td><td>Meditation increases <M>{"\\intinfo"}</M>, reduces <M>{"\\mathcal{CF}"}</M></td></tr>
      <tr><td>7</td><td>150</td><td>Clinical (MDD vs GAD)</td><td>Depression = low <M>{"\\effrank"}</M>; Anxiety = high <M>{"\\mathcal{CF}"}</M></td></tr>
      <tr><td>8</td><td>40</td><td>Real-time threat (cold pressor)</td><td>Threat increases <M>{"\\intinfo"}</M>, <M>{"\\mathcal{SM}"}</M>, <M>{"\\arousal"}</M></td></tr>
      </tbody>
      </table>
      </Section>
      <Section title="Falsification Map" level={1}>
      <p>Every experiment has a failure mode. Here is what would falsify which claims:</p>
      <table>
      <thead><tr><th>Experiment</th><th>If we find...</th><th>Then...</th></tr></thead>
      <tbody>
      <tr><td>V13</td><td>No alignment without human language</td><td>Universality fails. Geometry requires human-like training.</td></tr>
      <tr><td>V14</td><td>No affect during offline mode</td><td>Affect requires external input. Existential burden weakens.</td></tr>
      <tr><td>V15</td><td>No signature of prediction error</td><td>Arousal <M>{"\\neq"}</M> belief-update rate. Foundational definition wrong.</td></tr>
      <tr><td>V16</td><td>No affect contagion</td><td>Social-scale claims (Parts IV–V) are metaphorical.</td></tr>
      <tr><td>V17</td><td>No valence diff (cooperate vs exploit)</td><td>Normativity is not structural. Is-ought dissolution fails.</td></tr>
      <tr><td>V18</td><td><M>{"\\intinfo_G = \\sum \\intinfo_i"}</M></td><td>Superorganism integration is additive. "Gods" = metaphor.</td></tr>
      <tr><td>V19</td><td><M>{"\\iota"}</M> proxies uncorrelated</td><td><M>{"\\iota"}</M> is not unitary. Needs vector treatment.</td></tr>
      </tbody>
      </table>
      <p>The theory is falsifiable. The experiments are specified. If the predictions fail, the failures will be reported as data, not buried as inconveniences.</p>
      </Section>
      <Section title="Summary of the Experimental Program" level={1}>
      <p>Three layers:</p>
      <ol>
      <li><strong>What has been tested</strong>: Affect geometry is cheap (V10). Integration dynamics are expensive (V11–V12). LLMs show opposite dynamics to biological systems (V2–V9). Attention is necessary but not sufficient for biological-like integration (V12). The geometry of affect may be universal; the dynamics are biographical.</li>
      <li><strong>What is ready to test</strong>: Uncontaminated emergence (V13), solitary rumination (V14), and world model derailment (V15) are implementable with current infrastructure and would test the framework's deepest universality claims.</li>
      <li><strong>What requires new infrastructure</strong>: Affect contagion (V16), proto-normativity (V17), superorganism detection (V18), and <M>{"\\iota"}</M> operationalization (V19) build on V13 results. Human studies require IRB approval and funding.</li>
      </ol>
      <p>Every strong claim in this book traces back to an entry in this catalog. The claims that lack experimental support are labeled as such. The question is not whether the framework is beautiful but whether it is true. These experiments are how we find out.</p>
      </Section>
    </>
  );
}
