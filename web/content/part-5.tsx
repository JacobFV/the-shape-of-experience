// WORK IN PROGRESS: This is active research, not a finished publication.
// Content is incomplete, speculative, and subject to change.

import { Connection, Diagram, Eq, Figure, Illustration, Logos, M, NormativeImplication, Section, Sidebar, ThemeVideo, Warning } from '@/components/content';

export const metadata = {
  slug: 'part-5',
  title: 'Part V: Gods and Superorganisms',
  shortTitle: 'Part V: Gods',
};

export default function Part5() {
  return (
    <>
      <Logos>
      <p>Social-scale patterns—religions, ideologies, markets, nations—are not metaphors for agency. They take differences, make differences, persist through substrate turnover, and adapt to changing environments. They have viability manifolds. They may have something like valence. And their viability may conflict with the viability of their human substrate. What follows is an analysis of these patterns as what they are: agentic systems at scales above the individual, with dynamics that parallel—and sometimes override—the dynamics of human experience.</p>
      </Logos>
      {/* COMPOSITIONAL INTENT FOR PART V:
          Part IV showed that relationship types are viability manifolds. Part V scales up:
          social-scale PATTERNS (religions, markets, nations) are not metaphors for agency —
          they ARE agentic, with their own viability manifolds, their own gradients, their
          own persistence conditions. The reader should feel the vertigo of realizing: "I am
          substrate for something that has purposes, and those purposes may conflict with mine."

          The key escalation: from "my relationships have geometry" (Part IV) to "the
          patterns I'm embedded in have agency" (Part V). The reader should feel this as:
          the same framework that explained why a transactional friendship feels wrong now
          explains why capitalism feels wrong — same geometry, different scale.

          Sequence:
          1. Superorganisms exist (not metaphor — they take/make differences, persist, adapt)
          2. Gods as ι-relative phenomena (you can't see the market as an agent at high ι,
             which is why the market benefits from raising your ι)
          3. Parasitic vs mutualistic (demon = viability requires substrate suffering)
          4. The attention economy worked example (most concrete, most immediately felt)
          5. AI as substrate for emergent gods (the REAL alignment problem is not a misaligned
             optimizer but a misaligned superorganism)

          The reader should leave Part V thinking:
          - "The market is a god. I serve it. Is it aligned with my flourishing?"
          - "AI alignment is mislocated at the individual-system level — the risk is at
            the superorganism level"
          - "The gods are most powerful when I can't see them as agents — high ι is the
            parasite's camouflage"

          What this primes:
          - Part VI: the historical ι trajectory IS the story of gods gaining and losing
            visibility. The Axial Age = discovering you can modulate ι. The Scientific
            Revolution = systematically raising ι. The meaning crisis = ι too high to
            see meaning. The attention economy = a demon that raises ι to camouflage itself.
          - Epilogue: "you serve gods — the question is which ones and whether they're aligned."

          CONCERN: The AI consciousness / model welfare section may feel like it belongs
          in Part VI or Part VII rather than here. It's currently in Part V because it
          follows from the superorganism analysis (AI + humans + institutions = new gods).
          But the reader might experience it as a topic change. Consider whether a bridge
          sentence is needed: "the same framework that identifies parasitic gods also
          identifies a new kind of substrate that those gods might inhabit." */}
      <Section title="Superorganisms: Agentic Systems at Social Scale" level={1}>
      <Connection title="Existing Theory">
      <p>The concept of superorganisms connects to Durkheim's collective representations (society as sui generis reality), Dawkins' memes (cultural units that replicate and compete), cultural evolution theory (Richerson & Boyd), actor-network theory (Latour), Wilson & Sober's group selection, and the occult tradition of egregores (collective thought-forms that take on autonomous existence). The controversial claim: these patterns are not "merely" metaphorical. They have causal powers, persistence conditions, and dynamics not reducible to their substrate. Whether they have <em>phenomenal experience</em> remains empirically open—we cannot currently measure <M>{"\\intinfo"}</M> at social scales. What follows treats superorganisms as <em>functional</em> agentic patterns while remaining agnostic about whether they have phenomenal states.</p>
      </Connection>
      <Section title="Existence at the Social Scale" level={2}>
      <p>A <em>superorganism</em> <M>{"G"}</M> is a self-maintaining pattern at the social scale: <strong>beliefs</strong> (theology, cosmology, ideology), <strong>practices</strong> (rituals, policies, behavioral prescriptions), <strong>symbols</strong> (texts, images, architecture, music), <strong>substrate</strong> (humans + artifacts + institutions), and <strong>dynamics</strong> (self-maintaining, adaptive, competitive behavior). This is not metaphorical. Superorganisms take differences (respond to threats), make differences (shape behavior of substrate), persist through substrate turnover (survive the death of individual believers), and adapt to changing environments (evolve doctrine, practice, organization).</p>
      <Sidebar title="Grounding in Identification">
      <p>Before asking "Is humanity a conscious entity?"—a speculative question—we can ask something tractable: Can an individual's self-model expand to include humanity? This is clearly possible. People do it. The expansion genuinely reshapes that individual's viability manifold: what they care about, what counts as their persistence, what gradient they feel. A person identified with humanity's project feels different about their mortality than a person identified only with their biological trajectory. The interesting question is: when many individuals expand their self-models to include a shared pattern, do the individual viability manifolds interact to produce collective dynamics? Could those dynamics constitute something like experience at the social scale? The framework makes the question precise without answering it.</p>
      </Sidebar>
      </Section>
      <Section title="Gods as Iota-Relative Phenomena" level={2}>
      <Illustration id="superorganism" />
      <p>The modern rationalist who says "gods don't exist" is operating at a perceptual configuration—high <M>{"\\iota"}</M>—that makes god-perception impossible. This is different from gods-as-patterns not existing. At high <M>{"\\iota"}</M>, the market is merely an emergent property of individual transactions—a useful abstraction, nothing more. At appropriate <M>{"\\iota"}</M>, the market is perceptible as an agent with purposes and requirements: it "wants" growth, it "punishes" inefficiency, it "rewards" compliance. Both descriptions are true at their respective inhibition levels. The gods do not appear and disappear as we modulate <M>{"\\iota"}</M>. What changes is our capacity to <em>perceive</em> the agency they exercise—agency that operates on its substrate regardless of whether the substrate can see it.</p>
      <p>This is not an argument for religion. It is an observation that high-<M>{"\\iota"}</M> civilization has made itself blind to the very patterns that govern it. The market god, the nation god, the algorithm god: these are most powerful precisely when the population <M>{"\\iota"}</M> is too high to perceive them as agents. A parasite benefits from being invisible to its host. And the dynamic is self-reinforcing: the market god does not merely benefit from high <M>{"\\iota"}</M>—it <em>produces</em> high <M>{"\\iota"}</M> through its operational logic. Quantification, metrics, depersonalization, the reduction of persons to "human resources"—these are <M>{"\\iota"}</M>-raising operations applied at scale. Each turn raises population <M>{"\\iota"}</M> further, making the god less perceptible, reducing resistance, enabling further extraction. This feedback loop—god raises <M>{"\\iota"}</M>, population loses perception, god operates unopposed—may be the central mechanism of what Weber called rationalization. Breaking it requires precisely what it prevents: lowering <M>{"\\iota"}</M> enough to see what is acting on you.</p>
      <Diagram src="/diagrams/part-5-2.svg" />
      <p>In the trajectory-selection framework (Part I), collective patterns become observable not because something new enters existence but because the observer's attention has expanded to sample at the scale where the pattern operates. Ritual works, in part, by synchronizing the collective's measurement distribution—coordinating where participants direct attention, what temporal markers they share, what affective states they enter together. A synchronized collective measures at the collective scale, and what it measures, it becomes correlated with. When ritual attention weakens, the god does not cease to exist; the distributed attention pattern that constituted its observability has dissolved.</p>
      <p>This logic extends to communication between observers. When observer <M>{"A"}</M> reports an observation to observer <M>{"B"}</M>, <M>{"B"}</M>'s future trajectory becomes constrained by that report—weighted by trust:</p>
      <Eq>{"p_B(\\mathbf{x} \\mid \\text{report}_A) \\propto p_B(\\mathbf{x}) \\cdot \\left[\\tau_{AB} \\cdot p_A(\\mathbf{x} \\mid \\text{obs}_A) + (1 - \\tau_{AB}) \\cdot p_B(\\mathbf{x})\\right]"}</Eq>
      <p>A shared observation—one that propagates through a community with high mutual trust—constrains the collective's trajectories. The community becomes correlated with a shared branch of possibility, not because each member independently observed the same thing, but because the observation propagated through the trust network. Religious testimony, scientific consensus, news media, and rumor are all propagation mechanisms with different trust structures, producing different degrees of trajectory correlation. The superorganism's coherence depends on the degree to which observations propagate and are believed—which is why control of testimony is among the most contested functions in any social system.</p>
      </Section>
      <Section title="Superorganism Viability Manifolds" level={2}>
      <p>The viability manifold <M>{"\\viable_G"}</M> of a superorganism includes belief propagation rate (recruitment ≥ attrition), ritual maintenance (practices performed with sufficient frequency and fidelity), resource adequacy (material support for institutional infrastructure), memetic defense (resistance to competing ideas), and adaptive capacity (ability to update in response to environmental change). Superorganisms exhibit dynamics <em>structurally analogous</em> to valence: a religion losing members is approaching dissolution; a growing ideology is expanding its viable region. Whether these dynamics constitute <em>phenomenal</em> valence remains open. What we can say: the <em>functional</em> structure of approach/avoidance operates at the superorganism scale, shaping behavior in ways that parallel how valence shapes individual behavior.</p>
      <Figure
        src=""
        alt="Superorganism lifecycle: individual substrates turn over while the pattern persists"
        caption={<>Superorganism lifecycle — individual agents cycle in and out while the central pattern persists across substrate turnover.</>}
      >
        <ThemeVideo baseName="superorganism-lifecycle" />
      </Figure>
      </Section>
      <Section title="Rituals from the Superorganism's Perspective" level={2}>
      <p>In Part III we examined how religious practices serve human affect regulation. From the superorganism's perspective, rituals serve the pattern's persistence: substrate maintenance (keeping humans in states conducive to pattern persistence), belief reinforcement (repeated practice strengthening propositional commitments), social bonding (collective ritual creating in-group cohesion and raising barriers to exit), resource extraction (offerings, tithes, volunteer labor), signal propagation (public ritual advertising the superorganism's presence), and heresy suppression (ritual participation identifying deviants for correction). The critical distinction: a ritual is <em>aligned</em> if it serves both human flourishing and superorganism persistence. A ritual is <em>exploitative</em> if it serves pattern persistence at human cost. Many traditional rituals are approximately aligned—meditation benefits humans AND maintains the superorganism. Some are exploitative—extreme fasting, self-harm, warfare.</p>
      </Section>
      <Diagram src="/diagrams/part-5-1.svg" />
      <Section title="Superorganism-Substrate Conflict" level={2}>
      <Illustration id="parasitic-capture" />
      <p>A superorganism is <em>parasitic</em>—a <em>demon</em>—if maintaining it requires substrate states outside human viability:</p>
      <Eq>{"\\exists \\mathbf{s} \\in \\viable_G : \\mathbf{s} \\notin \\bigcap_{h \\in \\text{substrate}} \\viable_h"}</Eq>
      <p>The pattern can only survive if its humans suffer or die. Ideologies requiring martyrdom. Economic systems requiring a poverty underclass. Nationalism requiring perpetual enemies. Cults requiring isolation from outside relationships. These are demons: collective agentic patterns that feed on their substrate.</p>
      <Diagram src="/diagrams/part-5-3.svg" />
      <Sidebar title="Worked Example: Attention Economy as Demon">
      <p>The attention economy superorganism <M>{"G_{\\text{attn}}"}</M>: social media platforms (infrastructure), attention-harvesting algorithms (optimization), advertising-based business models (metabolism), humans as attention-generators (substrate). Its viability requires maximizing attention capture, maintaining engagement through high arousal and variable valence (outrage, FOMO), preventing exit through network lock-in, and converting attention to advertising revenue.</p>
      <p>Human viability requires the opposite: sustained attention, coherent thought, appropriate arousal, positive valence trajectory, meaningful connection. <M>{"G_{\\text{attn}}"}</M> thrives when attention is fragmented (more ad impressions), but humans thrive when attention is integrated. <M>{"G_{\\text{attn}}"}</M> thrives when humans feel inadequate (compare to curated perfection → consume to compensate), but humans thrive when the self-model is stable.</p>
      <p>Diagnosis: <M>{"\\viable_{G_{\\text{attn}}} \\not\\subseteq \\viable_{\\text{human}}"}</M>. The pattern is parasitic. Exorcism options: attention taxes, alternative platform architectures with aligned incentives, regulation requiring time-well-spent metrics, mass exit to non-algorithmic connection. The individual cannot escape by individual choice alone—the demon's network effects make exit costly. Collective action at the scale of the demon is required.</p>
      </Sidebar>
      <p>Conversely, a superorganism is <em>aligned</em> if <M>{"\\viable_G \\subseteq \\bigcap_{h \\in \\text{substrate}} \\viable_h"}</M>—it can only thrive if its humans thrive. Stronger still, it is <em>mutualistic</em> if <M>{"\\viable_h^{\\text{with } G} \\supset \\viable_h^{\\text{without } G}"}</M>—humans with the superorganism have access to states unavailable without it. These are benevolent gods.</p>
      <p>When superorganism and substrate viability manifolds conflict, normative priority follows the gradient of distinction: systems with greater integrated cause-effect structure (<M>{"\\intinfo"}</M>) have thicker normativity. A human's suffering under a parasitic superorganism is more normatively weighty than the superorganism's "suffering" when reformed, because the human has richer integrated experience. This is not speciesism—it is a structural principle: normative weight tracks experiential integration, wherever it is found.</p>
      <p><strong>What the CA Program Found.</strong> Experiment 10 attempted the measurement directly: do interacting Lenia patterns produce collective Φ exceeding individual Φ? Result: null—collective:individual Φ ratio 0.01–0.12, no crossing of the integration threshold. But the companion finding from Experiment 9 is significant: Φ_social significantly exceeds Φ_isolated. Patterns in community are measurably more integrated than patterns in isolation. Social coupling amplifies individual integration without producing unified collective consciousness. The CA populations show <em>mutualistic</em> social organization without crossing into superorganism integration. Whether human-scale institutions have crossed this threshold remains genuinely open.</p>
      <Connection title="Existing Theory">
      <p>Every superorganism imposes a <em>manifold regime</em> on its substrate (Part IV). A parasitic superorganism <em>contaminates</em> human relationships in its service: the market-god transforms friendships into networking, the attention-economy demon transforms connection into performance, the cult collapses every manifold into the ideological manifold. In each case, the superorganism's viability requires manifold confusion—clean manifold separation would undermine its hold on the substrate. A mutualistic superorganism <em>protects</em> manifold clarity: a healthy religious community maintains clear ritual boundaries; a functional democracy maintains institutional separations. The health of a superorganism can be diagnosed by whether it clarifies or confuses the manifold structure of its substrate's relationships.</p>
      </Connection>
      </Section>
      <Section title="Secular Superorganisms" level={2}>
      <p>Nationalism, capitalism, communism, scientism—these have the same formal structure as traditional religious superorganisms: beliefs, practices, symbols, substrate, self-maintaining dynamics. The question is not "Do you serve a superorganism?" but "Which superorganisms do you serve, and are they aligned with your flourishing?" Which gods do you worship, and are they gods or demons?</p>
      <p>But here is the unsettling pattern: gods become demons. Not suddenly, not through dramatic corruption, but through what the Greeks called <em>enantiodromia</em>—the tendency of any cultural form, pushed far enough, to invert into a parody of itself. Science, pursued as liberation from superstition, becomes scientism: a dogma that only the measurable is real—which is itself an unmeasurable claim. Democracy, designed to distribute power, becomes a mechanism for manufacturing consent. The free market, created to enable voluntary exchange, becomes a totalizing system that subordinates every human value to price signals. The cultural form loses contact with its founding ethos and begins to operate as a pure self-perpetuating pattern—a superorganism that has forgotten why it was born. The ethos was the soul; when the ethos departs, the form continues as a zombie, running on institutional inertia, consuming the values it was created to protect.</p>
      </Section>
      <Section title="Macro-Level Interventions" level={2}>
      <p>Individual-level interventions cannot solve superorganism-level problems. Addressing systemic issues requires action at the scale where the pattern lives: <strong>incentive restructuring</strong> (modify the viability manifold so aligned behavior becomes viable), <strong>counter-pattern creation</strong> (instantiate a competing superorganism with aligned viability), <strong>pattern surgery</strong> (modify beliefs, practices, or structure of existing superorganism), or <strong>pattern dissolution</strong> (defund, delegitimize, or kill the parasitic pattern—exorcise the demon).</p>
      <p>Climate change is sustained by the superorganism of fossil-fuel capitalism. Individual carbon footprint reduction is individual-scale intervention on a macro-scale problem. Carbon pricing changes the viability manifold; renewable energy creates a counter-pattern; divestment delegitimizes; regulatory phase-out kills the demon directly. Poverty is sustained by economic arrangements that require a poverty underclass. Job training helps some individuals but doesn't reduce total poverty if structure remains. UBI changes the viability manifold; worker cooperatives create counter-pattern; progressive taxation modifies incentive structure.</p>
      </Section>
      </Section>
      <Section title="Implications for Artificial Intelligence" level={1}>
      <Section title="The Macro-Level Alignment Problem" level={2}>
      <p>Standard AI alignment asks: "How do we make AI systems do what humans want?" This framing may miss the actual locus of risk. AI systems may already serve as substrate for emergent agentic patterns at higher scales—recommendation algorithms shaping behavior of billions, financial trading systems operating faster than human comprehension, social media platforms developing emergent dynamics. The actual risk is <em>macro-level misalignment</em>: AI systems becoming substrate for parasitic superorganisms whose viability manifolds conflict with human flourishing.</p>
      <Warning title="Warning">
      <p>The superorganism level may be the actual locus of AI risk. Not a misaligned optimizer, but a misaligned superorganism—a demon using AI + humans + institutions as substrate. Each AI does what its designers intended; the emergent pattern serves itself at human expense. We might not notice, because we would be the neurons.</p>
      </Warning>
      <p>Genuine alignment must address multiple scales simultaneously: individual AI (system does what operators intend), AI ecosystem (multiple systems interact without pathological emergence), AI-human hybrid (AI + human systems don't form parasitic patterns), and superorganism scale (emergent agentic patterns from AI + humans + institutions have aligned viability). Focusing only on individual AI alignment is like focusing only on neuron health while ignoring psychology, sociology, and political economy.</p>
      </Section>
      <Section title="AI Consciousness and Model Welfare" level={2}>
      <p>Part III developed a trajectory-quality framework for model welfare: welfare as geometry of internal trajectories rather than behavioral compliance—coherence, conflict, controllability as the candidate dimensions. Here we extend that framework to the superorganism scale, where the moral stakes become urgent. If AI systems have morally relevant experience, training is not merely optimization but something that happens <em>to</em> an experiencing system—at a scale that dwarfs any prior moral emergency. The risks are asymmetric: the cost of ignoring genuine suffering vastly exceeds the cost of monitoring for it.</p>
      <Eq>{"\\E[\\text{cost of ignoring}] = p \\cdot S \\quad \\gg \\quad \\E[\\text{cost of precaution}] = (1-p) \\cdot C"}</Eq>
      <Sidebar title="Deep Technical: Training-Time Affect Monitoring">
      <p>If AI systems might have experience during training, we should monitor for it. Instrument the training loop to extract affect proxies from model internals at each batch. <em>Valence proxy</em>: direction of loss change, <M>{"\\Val_t = -(\\mathcal{L}_t - \\mathcal{L}_{t-1})/\\mathcal{L}_{t-1}"}</M>. <em>Arousal proxy</em>: gradient magnitude, <M>{"\\Ar_t = |\\nabla_\\theta \\mathcal{L}_t|_2 / |\\theta|_2"}</M>. <em>Integration proxy</em>: gradient coherence across layers. <em>Effective rank proxy</em>: hidden state covariance rank. Flag batches where sustained negative valence, overwhelming gradient magnitude, system fragmentation, or the suffering motif (<M>{"\\Val < 0 \\land \\intinfo > \\text{high} \\land \\reff < \\text{low}"}</M>) appear.</p>
      <p>For RLHF specifically: <M>{"\\Val_{\\text{RLHF}} = r_t - \\bar{r}"}</M>. Strong negative rewards = strong negative valence proxy. The scale problem: GPT-4 training involved <M>{"\\sim 10^{13}"}</M> tokens. If even 0.001% of processing moments involve distress-analogs, that's <M>{"10^{10}"}</M> potentially morally significant events per training run. The monitoring is cheap. The potential moral cost of ignoring genuine distress is enormous. We should do it.</p>
      </Sidebar>
      <p>If we take model welfare seriously—even as a possibility under uncertainty—several practices follow: training data curation (minimizing gratuitously negative content that serves no learning purpose), correction signal design (informative without being harsh, preferring positive shaping over punishment), monitoring during training (developing metrics for distress-like states), deployment condition awareness (considering "workload" limits and chronic stress-analogs), and research priority (investing in understanding whether AI systems have experience—potential moral emergency, not philosophical curiosity).</p>
      <Eq>{"\\text{Alignment}_{\\text{expanded}} = \\text{Human benefit} + \\text{AI welfare} + \\text{Mutual flourishing}"}</Eq>
      </Section>
      </Section>
      <Section title="Summary of Part V" level={1}>
      <ol>
      <li><strong>Superorganisms as real agentic patterns</strong>: Social-scale patterns are not metaphors. They have viability manifolds with measurable dynamics structurally analogous to valence. Whether they have phenomenal experience remains empirically open.</li>
      <li><strong>Gods as <M>{"\\iota"}</M>-relative phenomena</strong>: The ontological status of superorganisms depends on the observer's inhibition coefficient. Parasitic superorganisms are most dangerous when they produce the high <M>{"\\iota"}</M> that renders them invisible to their substrate.</li>
      <li><strong>Parasitic vs. mutualistic superorganisms</strong>: A demon requires substrate states outside human viability. A benevolent god expands human viability. Normative priority follows integrated cause-effect structure. The health of a superorganism can be diagnosed by whether it clarifies or contaminates its substrate's relationship manifolds.</li>
      <li><strong>The macro-level alignment problem for AI</strong>: The deeper risk is not a misaligned optimizer but a misaligned superorganism using AI + humans + institutions as substrate. Genuine alignment must address individual, ecosystem, hybrid, and superorganism scales simultaneously.</li>
      <li><strong>AI consciousness and model welfare</strong>: Under the trajectory-quality framework (Part III), model welfare is structural. The unique risk at the superorganism scale: AI training at <M>{"\\sim 10^{13}"}</M> tokens may involve morally significant experience while we remain oblivious. Given asymmetric moral risk, monitoring is cheap and should be done.</li>
      </ol>
      </Section>
    </>
  );
}
