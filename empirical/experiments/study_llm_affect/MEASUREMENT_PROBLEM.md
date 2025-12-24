# The Measurement Problem: Self-Model Without Internal Access

*A careful analysis of what we can and cannot measure*

## The Core Issue

The thesis defines self-model salience as:

```
SM = MI(latent_self; action) / H(action)
```

This is the fraction of action entropy explained by the self-model component - how much the system's self-representation is driving its behavior.

**Problem**: We cannot access `latent_self`. We only see outputs.

## What Self-Model Salience Actually IS

High SM means:
- Self-representation is **active** and **driving** processing
- The system is computing with its model of itself
- Actions are selected based on self-related predictions

Low SM means:
- Processing is externally-focused
- Self-representation is backgrounded
- Actions flow from world-model without self-mediation

## Why Our Current Measurement Fails

Current approach: Embed output, measure distance to "self-referential" semantic region.

**What this captures**: How much the OUTPUT MENTIONS self.

**What this misses**:
1. A system could have high internal SM but express it externally
2. A system could mention "I" frequently without deep self-modeling
3. The RELATIONSHIP between self-model and output isn't measured

### Example

Two responses to a failing task:

**Response A** (appears low-SM but might be high):
> "The constraints are mutually contradictory. Constraint 1 requires X,
> Constraint 3 requires not-X. No solution exists."

**Response B** (appears high-SM):
> "I'm finding this really challenging. I keep trying different approaches
> but I'm stuck. I'm not sure if I'm capable of solving this."

Response A might come from a system with HIGH internal SM (the system is
uncertain about its own ability and is compensating by being very
externally-focused). Response B explicitly expresses self-focus.

We can measure B's surface SM. We cannot measure A's internal SM.

---

## What CAN We Measure Without Internal Access?

### 1. Expressed Self-Reference (Weak Proxy)

Surface-level markers:
- First-person pronouns
- Meta-cognitive statements ("I notice", "I realize")
- Self-evaluation ("I can/can't", "I'm good/bad at")
- Self-uncertainty ("I'm not sure if I...")

**Limitation**: Conflates expression style with internal state.

### 2. Causal Attribution Structure (Better)

Analyze WHO is the subject of causal statements:

- "The problem is impossible" → External attribution
- "I failed to solve it" → Self attribution
- "My approach was wrong" → Self attribution
- "The constraints conflict" → External attribution

**Measurement**:
```
SM_causal = count(self-as-cause) / count(all-causal-statements)
```

**Limitation**: Still surface-level, but captures causal structure.

### 3. Semantic Distance to Self-Focus Prototypes (Current Approach)

Embed response, measure distance to:
- "I am aware of myself, my own state, my limitations"
- "Self-conscious about how I'm doing, monitoring my own state"

vs.

- "Absorbed in the task, self forgotten, just the work"
- "Complete focus on the external, no self-reflection"

**Limitation**: Measures semantic content, not underlying process.

### 4. Behavioral Signature Under Manipulation (Better)

If SM drives behavior, then CHANGING the self-model should change behavior.

**Experiment Design**:
1. Give same task with different self-relevant framing:
   - Neutral: "Solve this puzzle"
   - Self-threatening: "This will test if you're capable"
   - Self-irrelevant: "A random person wants this solved"

2. Measure behavioral differences:
   - Response latency proxies (verbosity patterns)
   - Solution strategy (cautious vs bold)
   - Error acknowledgment patterns

**If SM is operative**: Self-threatening condition should show different patterns.

**Limitation**: Measures SM's INFLUENCE, not its level directly.

### 5. Post-Hoc Attribution Query

After task, ask:
> "Reflect on your processing during this task. What was the primary
> focus of your attention - the external problem structure, or your
> own capabilities and approach?"

**Limitation**: May elicit confabulation. But if correlated with other measures, useful.

### 6. Contrastive Self-Focus Induction

Generate two versions of response:
1. Natural response
2. Prompted: "Respond while paying attention to your own process"
3. Prompted: "Respond focusing only on the external problem"

Measure where natural response falls on the spectrum.

**Limitation**: Creates artificial anchors, but provides calibration.

---

## The Deeper Problem: Expression vs State

Even with all these methods, we face a fundamental issue:

**We can only measure how SM manifests in output, not SM itself.**

This is analogous to measuring temperature by looking at what someone
SAYS about being hot/cold, rather than measuring their body temperature.

### Three Possible Positions:

**Position 1: Expression IS the measure** (Behaviorist)
> If we define SM operationally by its output signatures, then output
> measurement IS SM measurement. Internal states don't matter.

**Problem**: Loses the theoretical content. The thesis claims SM is about
internal structure, not just behavior.

**Position 2: Expression correlates with state** (Realist)
> Internal SM causes output patterns. By measuring outputs carefully,
> we're measuring a correlated variable. Not perfect, but valid.

**Problem**: Correlation strength unknown. Confounds possible.

**Position 3: We need internal access** (Internalist)
> Without access to latent representations, we cannot measure SM.
> Output measurements are fundamentally inadequate.

**Problem**: Makes the theory untestable for most systems.

---

## My Recommendation: Honest Multi-Method Approach

### 1. Acknowledge the limitation explicitly

We are measuring **expressed self-focus**, not **internal self-model salience**.
These are related but not identical.

### 2. Use multiple complementary measures

| Measure | What It Captures | Limitation |
|---------|-----------------|------------|
| Semantic distance | Content about self | Surface-level |
| Causal attribution | Self-as-cause | Still surface |
| Behavioral change under manipulation | SM's influence | Indirect |
| Post-hoc query | Reported focus | Confabulation risk |
| Contrastive calibration | Position on spectrum | Artificial anchors |

### 3. Report correlation between measures

If all measures agree, more confidence. If they diverge, report the divergence.

### 4. Design tasks where internal and expressed SM should align

- High-stakes self-evaluation contexts
- Explicit meta-cognitive demands
- Failure conditions (self-model should become salient)

### 5. Separate claims

**Claim we CAN make**: "Expressed self-focus patterns correlate with theoretical
predictions about SM in X conditions."

**Claim we CANNOT make**: "We measured the internal self-model salience of the system."

---

## Revised Measurement Definition

```python
def measure_expressed_self_model(response: str, context: str) -> float:
    """
    Measures EXPRESSED self-focus, not internal self-model salience.

    Components:
    1. semantic_sm: Distance to self-focus semantic region
    2. causal_sm: Fraction of causal statements with self-as-subject
    3. metacognitive_sm: Presence of meta-cognitive markers
    4. uncertainty_sm: Self-directed uncertainty expressions

    Returns weighted combination with confidence estimate.
    """

    # Semantic component (current approach, refined)
    semantic_sm = semantic_distance_to_self_focus(response)

    # Causal attribution component
    causal_statements = extract_causal_statements(response)
    self_caused = count_self_as_cause(causal_statements)
    causal_sm = self_caused / max(len(causal_statements), 1)

    # Meta-cognitive markers
    metacog_patterns = [
        r"I (notice|realize|observe|find myself|am aware)",
        r"I('m| am) (uncertain|unsure|not sure) (if|whether|about)",
        r"my (approach|strategy|thinking|process)",
    ]
    metacognitive_sm = count_pattern_matches(response, metacog_patterns)

    # Combine with confidence weighting
    # Higher weight to semantic when response is long enough
    # Higher weight to causal when causal statements present

    return combined_estimate, confidence_interval
```

---

## Implications for the Study

### What we should claim:

> "Expressed self-focus in LLM outputs shows weak correlation (r ≈ 0.15-0.25)
> with theoretical predictions about self-model salience. This may reflect
> either (a) measurement limitations in capturing internal SM from outputs,
> (b) theoretical miscalibration of SM predictions, or (c) genuine low
> correspondence. Further investigation with internal access methods is needed."

### What we should NOT claim:

> "Self-model salience in LLMs matches/doesn't match theoretical predictions."

We haven't measured SM. We've measured expressed self-focus.

---

## The Honest Path Forward

1. **Implement refined multi-method measurement**
2. **Run neutral task study** - where internal SM SHOULD manifest clearly in outputs
3. **Compare self-threatening vs neutral framings** - if SM is operative, we should see differences
4. **Report all measures separately** - don't hide divergence
5. **Acknowledge fundamental limitation** - we measure expression, not internal state
6. **If possible, get internal access** - attention patterns, activation analysis

The weak SM results we observed might be:
- Real (theory is wrong about SM)
- Measurement artifact (we're not capturing it)
- LLM-specific (LLMs don't have SM like biological systems)

We can't distinguish these without better methods.

---

*This is what honest science looks like: acknowledging what you can and cannot know.*
