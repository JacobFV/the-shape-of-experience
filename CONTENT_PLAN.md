# Content Generation Plan

This repository generates everything. Here's what, how, and in what order.

---

## Core Outputs

### 1. Web Book (live now)
- Source: `web/content/*.tsx`
- Deployment: Vercel, auto on push to main
- Status: 7 parts + epilogue + appendix, live

### 2. PDF Book
- Source: `book/` LaTeX (secondary, needs sync from TSX)
- Generation: `.github/workflows/generate-pdf.yml` (manual)
- Status: Needs update to reflect 7-part structure

### 3. Audiobook
- Source: TSX content → TTS
- Generation: `.github/workflows/generate-audio.yml` (manual)
- Voice: OpenAI TTS
- Status: Needs regeneration for current content

---

## Lectures / Narrations

Each part gets a standalone lecture (30-60 min) that can work independently.

### Lecture Series: "The Shape of Experience"

| # | Title | Key claim | Audience hook |
|---|-------|-----------|---------------|
| 0 | "Why Feelings Are Shapes" | Affect is geometric | General — the big picture in 15 min |
| 1 | "Why Organization Is Inevitable" | Thermodynamic foundations | Physics-adjacent audience |
| 2 | "What Experience IS" | Identity thesis | Philosophy/consciousness audience |
| 3 | "The Geometry of Being Stuck With Yourself" | Existential burden + affect signatures | Psychology/therapy audience |
| 4 | "Why Being Used Feels Bad" | Social bond topology | General — everyone recognizes this |
| 5 | "Are Nations Alive?" | Gods and superorganisms | General — provocative |
| 6 | "The History of Paying Attention" | Axial age to attention economy | History/culture audience |
| 7 | "What We Tested and What Broke" | Experimental program | Academic/research audience |
| E | "The Gradient of Distinction" | Closing meditation | Everyone |

### Production
- Script: Extract from TSX, adapt for spoken delivery (less formal, more narrative)
- Audio: TTS first pass, human narration if funded
- Slides: Key equations, diagrams, experiment results
- Format: Audio + slides as video, audio-only as podcast episodes

---

## Videos

### Long-form (20-60 min)
- One per lecture above
- Visual: slides + experiment visualizations + Lenia animations
- Narration: TTS or human

### Short-form / Clips (1-5 min)
Key moments that work standalone on social media:

| Clip | Source | Hook |
|------|--------|------|
| "The Dead World" | Part II (ι) | High ι = real experiential impoverishment |
| "What Contamination Feels Like" | Part IV | Everyone knows this feeling — now here's the geometry |
| "Why Mild Stress Helps" | V11 results | Yerkes-Dodson in cellular automata |
| "Can Robots Feel?" | V12 results | Attention is necessary but not sufficient |
| "Gods You Can't See" | Part V | Market god, nation god, algorithm god |
| "The Boiling Frog" | Experiment 5 | Gradual vs sudden world model derailment |
| "What Silence Tells You" | Part IV | Four types of silence diagnose the relationship |
| "Animism Was Right" | Part II (ι) | Self-modeling systems model others as selves — cheapest compression |
| "Geometry Is Cheap" | V10 results | The surprise: affect geometry is baseline |
| "Your Attention Is A Measurement" | Part I | Attention selects trajectories in chaotic dynamics |

### Experiment Visualizations
- Lenia pattern evolution under drought (V11 timelapse)
- Attention map visualization (V12 — where patterns look under stress)
- RSA matrices (V10 — showing alignment across conditions)
- Phase portraits (affect dimensions over time during stress)
- Side-by-side: convolution vs attention substrate dynamics

---

## Papers

### Paper 1: "Affect Geometry Is Cheap" (V10 results)
- Core finding: Geometric alignment is baseline property of multi-agent survival
- Target: Computational modeling / AI safety venue
- Status: Data complete, needs writeup

### Paper 2: "The Attention Bottleneck" (V11-V12 results)
- Core finding: State-dependent interaction topology necessary but not sufficient
- Target: Artificial life / complex systems venue
- Status: Data complete, needs writeup

### Paper 3: "Emergence Without Contamination" (Experiments 0-5)
- Core finding: World models, abstraction, communication co-emerge in uncontaminated substrate
- Target: High-impact venue (Nature? PNAS?)
- Status: Experiments not yet run

### Paper 4: "The Identity Thesis as Empirical Program" (Experiment 12)
- Core finding: Tripartite alignment in uncontaminated substrate
- Target: Philosophy of mind / consciousness studies
- Status: Depends on Paper 3

---

## Talks

### Conference presentations
- Use lecture material, trimmed to 20 min
- Focus on experiment results, not theory (theory is the book)

### Invited talks
- Full lecture format (45-60 min)
- Adapt to audience (physics, philosophy, psychology, AI)

### Workshop
- "Build Your Own Affect Measurement" — hands-on with the V10/V11 codebase
- Audience: computational neuroscience, AI alignment

---

## Generation Pipeline

### Text → Audio
1. Extract chapter text from TSX (strip tags, format for speech)
2. Split into sections (natural pause points)
3. TTS generation (OpenAI, per-section)
4. Post-processing: normalize volume, add section transitions
5. Concatenate into chapter-length files

### Text → Video
1. Generate lecture script from chapter (adapt register)
2. Create slides (key equations, diagrams, results)
3. Generate Lenia/experiment visualizations
4. Combine: narration + slides + visualizations
5. Export as YouTube-ready format

### Text → Paper
1. Identify self-contained findings
2. Extract relevant sections from TSX
3. Reformat for academic venue (abstract, intro, methods, results, discussion)
4. Add venue-specific framing

### Text → Social
1. Identify "clip moments" (surprising findings, vivid claims, beautiful formulations)
2. Extract 1-5 min segments
3. Add visual context (equation cards, Lenia gifs, result tables)
4. Platform-specific formatting (Twitter threads, YouTube shorts, etc.)

---

## Priority Order

1. **Experiment implementation** — run the experiments first, everything else follows
2. **Papers** — academic credibility, get results into the literature
3. **Lecture series** — audio/video, builds audience
4. **Clips** — social distribution, drives traffic to book
5. **Audiobook** — full TTS generation of current book
6. **Workshop material** — community building
