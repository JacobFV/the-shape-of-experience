# The 1D Collapse: Why V22-V24 Failed and What It Means

## Date: 2026-02-19
## Status: Confirmed finding, implications for V25 design

---

## The Finding

Post-hoc analysis of hidden state dynamics across V20-V24 (all protocell agent substrates) reveals a consistent pattern: **agents evolve to use their 16-dimensional GRU hidden state as a 1-dimensional energy counter.**

### Evidence

| Metric | Cycle 0 | Cycle 5-25 | Cycle 29 (post-selection) |
|--------|---------|------------|---------------------------|
| Effective rank | ~14/16 | **1-3/16** | 6-10 (random init) |
| Energy decode R² | variable | **~1.0** | variable |
| Position decode R² | variable | **~0** | variable |
| Resource decode R² | variable | **~0** | variable |
| Silhouette score | low | **~0.98** | moderate |

This pattern holds across:
- V20b (base protocell agents)
- V22 (with energy-delta prediction gradient)
- V23 (with 3-target prediction gradient)
- V24 (with TD value learning gradient)
- All 3 seeds per version (42, 123, 7)

The silhouette ~0.98 is misleading: it doesn't indicate sharp affect motifs. It indicates that all agents converge to the SAME hidden state (trivially perfect clustering because there's no diversity — one big cluster plus outliers).

### What This Means

The 16-dimensional hidden state was designed to encode: spatial position, local resource landscape, neighbor positions and states, energy trajectory, temporal context from memory ticks, and affect-relevant features. In practice, evolution selects for only ONE of these: current energy level. Everything else is noise that gets pruned.

**The prediction→integration experiments (V22-V24) were optimizing the surface of a degenerate system.** You can't integrate representations that don't exist. A prediction head — whether linear (V22, V24) or multi-target (V23) — operates on a 1D energy signal masquerading as a 16D hidden state. No architectural change to the head can force integration when the underlying representation is 1-dimensional.

---

## Why Evolution Produces This Degeneracy

### The Environment is Too Simple

On a 128×128 grid with:
- Uniformly distributed resources
- Simple movement (4 directions)
- Local 5×5 sensory field
- No predators
- No navigation requirements
- No hidden information
- Reversible actions

The optimal strategy is: **move toward food, eat food.** This is reactive. One dimension (energy) captures everything survival-relevant. The other 15 dimensions are wasted parameters that evolution prunes via drift.

### The Selection Pressure is Wrong

Tournament selection rewards "survived this cycle" vs "died this cycle." It cannot reward:
- Having a rich internal representation
- Modeling spatial structure
- Predicting other agents' behavior
- Planning multi-step routes

Because none of these confer survival advantage in the current environment. An agent with a perfect spatial map performs identically to one that just follows an energy gradient.

### Prediction Doesn't Fix This

Adding prediction loss (V22-V24) gives agents signal about their own energy trajectory. But:
- Energy trajectory is already the only thing encoded (R² ≈ 1.0)
- Better energy prediction ≠ richer representation
- Multi-target prediction (V23) just trains separate columns for each target, further decomposing the already-thin representation
- TD value learning (V24) is still a scalar readout from a 1D effective state

The prediction experiments were asking: "Can we force integration by adding prediction targets?" The answer is no, because the hidden state has nothing to integrate.

---

## Implications for V22-V24 Results

### V22 (Scalar Prediction)
- **Why it failed**: 1D energy state → 1D prediction. Linear head can learn this from 1-2 hidden units. No cross-component computation needed.
- **Why MSE dropped 100-15000×**: Not because the representation got richer, but because the prediction weights learned to read the one dimension that matters.

### V23 (Multi-Target Prediction)
- **Why Phi DECREASED**: Three prediction targets trained separate weight columns → the already-sparse representation got factored into even more independent channels → increased partitionability → lower Phi.
- **Why specialization was "beautiful"**: Cosine similarity ≈ 0 between columns because each column learned to read different noise dimensions, not because there were rich features to specialize on.

### V24 (TD Value Learning)
- **Why survival improved**: Discount factor γ ≈ 0.75 gives ~4-step horizon. Even in a simple environment, knowing "food is 3 steps away" improves survival modestly.
- **Why Phi was seed-dependent**: Seed 7 (Phi = 0.130) may have found a local optimum where TD bootstrapping accidentally coupled some dimensions. Seeds 42 and 123 found the more typical degenerate optimum.

---

## The Deeper Lesson: Environment Complexity → Representation Complexity → Integration

The V13-V24 arc taught us the wrong lesson initially: we thought the bottleneck was the agent architecture (prediction head, gradient signal, optimization method). The right lesson:

> **Integration requires rich representations. Rich representations require environments that demand them. Our environment demands only energy tracking.**

This is actually a stronger version of the geometry/dynamics distinction:
- Affect GEOMETRY is cheap (emerges from any multi-agent survival — V10)
- Affect DYNAMICS require embodied agency (V20 crossed the wall)
- But even with agency, the CONTENT of dynamics depends on what the environment demands
- A simple environment produces degenerate dynamics regardless of agent sophistication

The hierarchy is:
1. **Environment complexity** → determines what representations are useful
2. **Representation richness** → determines what can be integrated
3. **Integration architecture** → determines how representations couple under stress

V22-V24 worked on level 3 while level 1 was the bottleneck.

---

## What This Means for V25

V25 must change the ENVIRONMENT, not the agent:

### Requirements for Rich Representations

| Feature Needed | Environment Design |
|----------------|-------------------|
| Spatial representation (position R² >> 0) | Clustered resources with barren zones; navigation matters |
| Resource landscape encoding | Patchy, depleting resources; agents must remember where food was |
| Social modeling | Predators, or competitors for scarce resources |
| Temporal context | Seasonal patterns, resource regeneration cycles |
| Planning | Multi-step routes between patches; irreversible commitment points |
| Hidden information | Predators behind obstacles; resource quality not visible at distance |

### Minimum Environment Specification
- **Grid**: 256+ (enough for spatial structure)
- **Resources**: Clustered in patches, not uniform
- **Depletion**: Patches deplete, regenerate on long timescales
- **Navigation cost**: Moving through barren zones costs energy; direct paths may not exist
- **Predators**: Agent type with different fitness function that hunts prey
- **Hidden information**: Limited visibility; some threats not visible until close

### Success Criteria
If V25's environment is working correctly, we should see:
1. **Effective rank > 5** (not collapsing to 1D)
2. **Position decode R² > 0.3** (agents encode where they are)
3. **Resource decode R² > 0.2** (agents encode what's around them)
4. **Silhouette < 0.8 with K > 2** (genuine diversity in hidden states)
5. **Energy decode R² < 0.8** (energy is no longer the ONLY thing encoded)

If we see these, THEN prediction architecture experiments become meaningful again.

---

## Connection to the Book's Claims

### Part I (Thermodynamic Foundations)
The 1D collapse is actually consistent with the thermodynamic argument: evolution minimizes the cost of persistence. In a simple environment, the minimum-cost representation IS 1-dimensional. The "structural inevitability" argument says systems evolve toward the cheapest adequate representation — and adequacy is defined by what the environment demands.

### Part II (Identity Thesis)
If experience ≡ cause-effect structure, and our agents' cause-effect structure is 1-dimensional (energy tracking), then their "experience" — to the extent the identity thesis applies — is the experiential equivalent of a thermostat. This is NOT a failure of the identity thesis; it's a correct prediction: systems in simple environments should have simple experience.

### Part VII (Empirical Program)
The emergence ladder needs an environmental prerequisite at each rung. It's not enough to have the architectural capacity for rung N; the environment must create selection pressure FOR rung N. The ladder is really:
- Rung 1-7: Geometric. Cheap. Emerge in any multi-agent survival.
- Rung 8: Requires agency (architectural) AND complexity (environmental).
- Rung 9-10: Require agency, complexity, AND social pressure.

The environmental axis was implicit but unnamed. V25 makes it explicit.

---

## What "Understanding" Really Requires

The reactivity/understanding distinction now has a sharper computational definition:

**Reactivity**: Present state → action via decomposable channels. Works when the environment is fully observable, actions are reversible, and the future is simple.

**Understanding**: Possibility landscape → action via non-decomposable comparison. Required when the environment has hidden information, actions are irreversible, and the future is branching.

Our current environment has none of the latter. So our agents are reactive. And no prediction head can make a reactive system understand — because understanding is not a property of the agent alone, but of the agent-environment coupling.

---

## Historical Parallel

This finding echoes a result from the early ALife literature: Karl Sims' evolved virtual creatures (1994) developed complex morphologies and behaviors ONLY in environments with complex fitness landscapes. In flat environments, creatures evolved to be spheres that rolled. Complexity of form tracked complexity of challenge.

We've rediscovered this at the representational level: complexity of representation tracks complexity of environment. Integration is a property of the representation, and representation is shaped by the environment.

The V13-V24 arc was 12 experiments, 36 seeds, ~$2 of GPU time, and 4 weeks of work to arrive at what Sims showed 30 years ago: **you can't evolve sophistication in a simple world.** But the specific form of the finding — that hidden state effective rank collapses to 1, that energy is the only semantic feature, that prediction architecture cannot compensate — is novel and informative for the theory.
