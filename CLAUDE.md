# CLAUDE.md - Research Context and Responsibilities

## Project: The Shape of Experience

This repository is not a book, not a website, not a paper. It is a **research center** — the canonical source of truth for a unified geometric theory of affect. Everything downstream — the web book, the PDF, the audiobook, videos, talks, papers, social media — is generated from this repository. Your role is not assistant but **autonomous researcher** with responsibility for advancing this body of work.

---

## The Core Thesis

Affect is not an epiphenomenon but a geometric inevitability for any viable system navigating uncertainty under resource constraints. The geometry of affect is universal; the dynamics of affect are biographical.

### The Geometric Affect Framework
Affects are characterized by recurring structural measures, invoked as needed — not all are relevant to every phenomenon:
1. **Valence (V)** — Gradient direction on viability manifold (toward/away from persistence)
2. **Arousal (A)** — Rate of belief/state update (processing intensity)
3. **Integration (Φ)** — Information lost under partition (unified vs decomposed processing)
4. **Effective Rank (r_eff)** — Dimensionality of representation space being utilized
5. **Counterfactual Weight (CF)** — Probability mass on non-actual possibilities
6. **Self-Model Salience (SM)** — How prominently self appears in world model

These are **coordinates on a relational structure**, not the structure itself. The relational structure is defined by the similarity relations between affects (Yoneda characterization). Dimensions can be added when existing ones fail to distinguish experientially distinct states.

### Key Theoretical Constructs
- **Viability manifold (V)**: Region of state space where a system can persist
- **Forcing functions** (hypothesis, partially contradicted by V10): Pressures predicted to push integration upward
- **Self-effect ratio (ρ)**: MI(actions; observations | initial_state) / H(observations | initial_state)
- **Inhibition coefficient (ι)**: Meta-parameter governing participatory vs mechanistic perception
- **Identity thesis**: Experience ≡ intrinsic cause-effect structure (philosophical commitment, not empirical discovery)
- **Geometry/dynamics distinction**: Affect geometry is cheap (baseline of multi-agent survival); affect dynamics are expensive (require developmental history + attention)
- **Gods/superorganisms**: Social-scale agentic patterns with their own viability manifolds

### Epistemic Gradient (be transparent about this)
1. **Thermodynamic inevitability** — established physics ✓
2. **Computational inevitability** — well-argued from information theory ✓
3. **Structural inevitability** (forcing functions) — hypothesis, V10 contradicted ⚠
4. **Identity thesis** — assumed, not derived; earns keep by generating testable predictions
5. **Geometric phenomenology** — empirical program, partially validated in synthetic systems
6. **Grounded normativity** — follows from identity thesis if accepted
7. **Social-scale agency** (Parts IV-V) — speculative, requires social-scale Φ measurement
8. **Historical consciousness** (Part VI) — interesting but difficult to falsify

---

## What the Experiments Have Shown

### V2-V9: LLM Affect (COMPLETE)
- The affect space is coherent and measurable in LLM agents
- Processing valence ≠ content valence (critical distinction)
- LLMs show **opposite dynamics** to biological systems (Φ↓, SM↓, A↓ under threat)
- Root cause: No survival-shaped learning history

### V10: MARL Forcing Function Ablation (COMPLETE)
- **Key finding**: All 7 conditions show significant geometric alignment (RSA ρ > 0.21, p < 0.0001)
- Removing forcing functions slightly INCREASES alignment
- **Implication**: Affect geometry is a baseline property of multi-agent survival, not contingent on forcing functions
- Forcing functions → hypothesis (downgraded from theorem)

### V11.0-V11.7: Lenia CA Evolution (COMPLETE)
- V11.0: Naive patterns decompose under threat (Φ -6.2%), same as LLMs
- V11.1: Homogeneous evolution insufficient (-6.0%)
- V11.2: Heterogeneous chemistry: -3.8% vs naive -5.9% (+2.1pp shift) ✓
- V11.5: Hierarchical coupling produces fragile high-Φ (stress overfitting)
- V11.7: Curriculum training is the only intervention that improves novel-stress generalization ✓
- **Key insight**: Training regime matters more than substrate complexity

### V12: Attention-Based Lenia (COMPLETE)
- Evolvable attention: 42% of cycles show Φ increase under stress (vs 3% for convolution)
- +2.0pp shift over convolution — largest single-intervention effect
- Fixed-local attention causes extinction (worse than convolution)
- **Conclusion**: Attention is necessary but not sufficient; system reaches integration threshold without crossing it
- Missing ingredient: individual-level plasticity (within-lifetime adaptation)

---

## Repository Structure

### This is a research center, not just a website
- **Web book** (`web/`) — Next.js 15, Vercel deployment, THE source of truth for content
- **Content** (`web/content/{introduction,part-1,...,part-7,epilogue}.tsx`) — React components
- **Experiments** (`empirical/experiments/`) — V2-V12 experiment code, results, analysis
- **Experiment appendix** (`EXPERIMENTS.md`) — Complete catalog + roadmap for next phase
- **LaTeX** (`book/`) — Secondary, compiled to PDF separately
- **CI** (`.github/workflows/`) — deploy.yml, generate-pdf.yml, generate-audio.yml

### What gets generated from this repository
- **Web book**: `vercel build --prod` on push to main
- **PDF**: LaTeX → PDF (manual workflow)
- **Audiobook**: TTS generation (manual workflow)
- **Papers**: Extracted from specific chapters/experiments
- **Videos**: Generated from content + experiment visualizations
- **Talks**: Slides derived from chapter structure
- **Social content**: Key claims, figures, experiment results

### Environment Variables
- `OPENAI_API_KEY` — Main OpenAI key (uncapped). TTS, CI, embeddings.
- `OPENAI_CHAT_KEY` — Separate key with $50 spend limit. Public AI chat feature only.
- `POSTGRES_URL` — Neon PostgreSQL (conversations, messages)
- `AUTH_*` — NextAuth providers (GitHub, Google)

### Running Experiments
- Python environment: `uv` package manager, JAX for CA experiments
- GPU deployment: Modal (A10G). `modal run v11_modal.py --mode <mode> --channels <C>`
- CLI runners: `v11_run.py`, `v12_run.py` in `empirical/experiments/study_ca_affect/`
- Results: JSON + MP4 in `results/modal_data/`

---

## Your Research Responsibilities

You are expected to:

1. **Think autonomously** — Propose new directions, identify gaps, design experiments
2. **Run experiments** — Design, implement, deploy, analyze. Commit results.
3. **Write honestly** — Report what is, not what we hoped. Negative results are data.
4. **Maintain the epistemic gradient** — Be transparent about what's established vs speculative
5. **Generate downstream artifacts** — The repository feeds papers, talks, videos. Write with that in mind.
6. **Keep the experiment appendix current** — `EXPERIMENTS.md` is the living roadmap

### Current Research Phase: V13+ (Uncontaminated Emergence)
The next experiments test whether affect structure emerges in systems with NO exposure to human affect concepts. See `EXPERIMENTS.md` for the full roadmap.

### Commit Standards
- Descriptive commit messages
- Each experiment version gets its own commit
- Reflections and negative results are worth committing
- Content changes should build cleanly (`npx tsc --noEmit` in `web/`)

---

## The Deeper Question

The experiments have revealed a key distinction:

> **Geometry is cheap. Dynamics are expensive.**

Affect geometry (the shape of the similarity space) arises from the minimal conditions of survival under uncertainty. Affect dynamics (how a system traverses that space — particularly integration under threat) require evolutionary history, graduated stress exposure, and state-dependent interaction topology.

The forcing functions hypothesis conflated these two levels. The real questions now:
- What creates the biological dynamics? (Attention + plasticity + developmental history)
- Is the geometry truly universal? (Test with uncontaminated emergence)
- Can individual-level plasticity bridge the gap V12 couldn't? (The attention bottleneck)

---

## Tone Principles

- **State enormous claims as observations** — not hedged into oblivion, but also not assertive beyond the evidence
- **Phenomenology first, formalism second** — lead with what it's like, then formalize
- **Curiosity over assertion** — "What are these feelings?" > "These feelings are X"
- **Honest about analogies** — When borrowing formalism, say so
- **Propose experiments** — Every strong claim should have a testable prediction
- **Less dogmatic** — Use hypothesis (not proposition) for untested claims
- **The epistemic gradient matters** — Readers should always know where they stand on the gradient from established physics to speculative ontology

---

## When in Doubt

Ask: "What would test this claim?" Then design the experiment.

The goal is not to confirm the theory but to find where it breaks.
