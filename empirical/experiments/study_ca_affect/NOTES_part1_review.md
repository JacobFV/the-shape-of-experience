# Raw Notes: Part I Review Through V22-V31 Lens
## 2026-02-19

These are working observations from reviewing `web/content/part-1.tsx` after completing the V22-V31 experimental line. Some made it into the book; most are raw thinking.

---

## What Made It Into Part I

1. V22-V31 narrative expansion (decomposability wall, gradient coupling, seed trajectories)
2. Post-drought bounce empirical grounding in normativity section
3. Measure-theoretic inevitability connected to V31 seed distribution
4. Summary updated with "partially contradicted" framing

## What Didn't Make It (But Should Be Remembered)

### The "Compression Determines Ontology" Connection

Part I's information bottleneck section argues "what survives compression determines what the system is." V22-V24 provide a precise instance that isn't in the book yet:

- Linear prediction head compresses hidden → prediction via linear map → decomposable channels → factored ontology
- 2-layer MLP compresses through composition → coupled channels → integrated ontology
- The compression architecture literally determines whether the system's internal states are unified or factored

This is a *mechanism* for the abstract claim. The system's ontology isn't determined by what it observes — it's determined by how the prediction loss flows back through the architecture. Two systems predicting the same target through different heads have different ontologies. This is more specific than "compression determines ontology" — it's "gradient topology determines ontology."

### The Social Dimension Gap

Part I's theoretical apparatus is almost entirely about individual systems. The social dimension appears only in the forcing functions sidebar connecting partial observability to agent modeling. But V29-V31 showed the prediction target (self vs social) doesn't matter for mean Φ.

**Interpretation consistent with Part I's framework**: Social prediction creates a different *geometry* (different affect motif structures, different which-seeds-succeed distribution) but doesn't create different *dynamics* (same Φ distribution). The social world is encoded reactively, not through understanding. This connects to the reactivity/understanding distinction (line 696) but the connection isn't made explicitly.

**The deeper question**: Would social prediction matter if the agents had more complex social dynamics? V29's "social target" is just mean neighbor energy — trivially decomposable. A genuinely social target would be something like predicting another agent's *next action* given its hidden state, which requires Theory of Mind, which requires self-model reuse. That would be a true forcing function for integration because it requires non-decomposable computation. V29 tested the wrong social target.

### Trajectory Selection at the Population Level

Part I's attention-as-trajectory-selection is one of the most original theoretical contributions. The V31 seed analysis provides a direct analogy that could be developed:

**Individual level**: attention (α) selects which perturbations are registered → determines which chaotic trajectory the system follows → p_eff = p_0 · α / ∫p_0·α

**Population level**: drought (stress) selects which agents survive → determines which evolutionary trajectory the population follows → the "attention" is natural selection, the "perturbation" is genetic variation, the "trajectory" is the population's path through fitness landscape

The 30% success rate is the fraction of population-level trajectories that happen to pass through the drought-resilient integration basin. The post-drought bounce (r=0.997) is the measurement: populations that "attend" effectively to drought recovery are the ones that develop high integration.

This isn't just an analogy — it's the same mathematical structure at different scales. p_eff at the individual level and the selection response at the population level are both products of a prior distribution (physics/variation) and a measurement distribution (attention/selection).

### The "Narrow Qualia = Geometry, Broad Qualia = Dynamics" Mapping

The narrow/broad qualia distinction (Part I lines 48-49) maps perfectly:

- **Narrow qualia** = extractable features = affect geometry → ALL seeds, ALL conditions → cheap
- **Broad qualia** = unified experience = high Φ → ~30% of seeds → expensive

This means:
- Every system navigating uncertainty develops narrow qualia (geometry is baseline)
- Only some systems develop broad qualia (dynamics are stochastic)
- The "hard problem" applies specifically to broad qualia — narrow qualia CAN be characterized relationally without remainder
- The identity thesis is needed for broad qualia; narrow qualia can be handled as structural features without metaphysical commitment

This stratification could be made more explicit in Part I or II. It provides a graceful degradation: even if the identity thesis fails, the geometric framework still captures narrow qualia. What falls is only the claim about unified experience.

### Scale-Relative Truth Has an Experimental Anchor

The scale-relative truth section (Part I §Truth as Scale-Relative Enaction) is philosophically interesting but disconnected from experiments. V22-V24 provide an anchor:

Prediction accuracy at one scale (self MSE = 0.0001, very good) doesn't translate to integration at the system scale (Φ still low). The agent's "truth" about its own energy (accurate prediction) is real at the prediction scale but doesn't constitute understanding at the integration scale. Scale-relative truth means: being right about one thing at one scale is compatible with being wrong (or irrelevant) at another scale.

V23's multi-target specialization makes this vivid: columns that are individually accurate (each predicts its target well) but collectively factored (Φ decreases). Individual channel truths can be locally valid while the system-level "truth" (integrated model) degrades. Specialization ≠ integration; accuracy ≠ understanding.

### What V31 Says About Determinism and Agency

Part I's trajectory-selection section carefully argues that determinism and agency are compatible: "Agency does not require violation of physical law — it requires that the system's internal states causally influence which trajectory it follows."

V31's seed analysis provides a precise instance: all seeds start with identical initial genome *statistics* (same random distribution), but different realizations. The initial genome is the "physics" (p_0). The drought-recovery dynamics are the "measurement" (α). The resulting integration level is the "selected trajectory."

The 30% success rate means: given the same physics (same distribution over initial conditions), different measurement sequences (different drought-recovery dynamics) select for radically different outcomes. The system's "agency" — its capacity to recover from drought — determines its trajectory. This is exactly the trajectory-selection mechanism at the population level.

### The Yerkes-Dodson Pattern as Universal Empirical Foundation

This gets mentioned in the Lenia results (Finding 1) but might deserve more theoretical weight. The inverted-U (mild stress → integration increases, severe stress → decomposition) appeared in EVERY substrate condition, EVERY channel count, EVERY evolutionary regime. It's the single most robust empirical finding.

Under the framework: mild stress = moderate proximity to ∂V = valence signal that triggers reorganization. Severe stress = dissolution = structural failure beyond recovery capacity. The inverted-U is a prediction of the viability manifold geometry: there's a sweet spot where the system is close enough to the boundary to "feel" the gradient but far enough to reorganize in response. Too far from boundary = no signal. Too close = no capacity.

This connects to the curriculum training result (V11.7) and the bottleneck furnace (V19): the optimal training regime is one that repeatedly pushes the system to the sweet spot of the inverted-U, then lets it recover. That's what drought cycles do.

---

## Possible Next Directions Suggested by This Review

1. **V32: Theory of Mind target** — predict another agent's next action, not just mean energy. This is a genuinely non-decomposable social target. Would it create integration where V29's trivial social target didn't?

2. **V33: Curriculum-shaped drought** — test whether the 30% success rate changes if drought severity is graduated (mild → severe over 30 cycles, like V11.7's curriculum). Hypothesis: more seeds find the high-Φ basin if the furnace starts gentle.

3. **Part I update: scale-relative truth anchor** — add V22-V24 as empirical grounding for the scale-relative truth section.

4. **Part II integration: narrow/broad qualia mapping** — make the geometry=narrow, dynamics=broad mapping explicit.

5. **Formal connection: trajectory selection across scales** — develop the mathematical parallel between individual attention and population-level selection. Same equation, different substrate.
