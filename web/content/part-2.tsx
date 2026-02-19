import { Align, Connection, Diagram, Eq, Experiment, Logos, M, OpenQuestion, Phenomenal, Proof, Section, Sidebar, TodoEmpirical } from '@/components/content';

export const metadata = {
  slug: 'part-2',
  title: 'Part II: The Identity Thesis and the Geometry of Feeling',
  shortTitle: 'Part II: Identity Thesis',
};

export default function Part2() {
  return (
    <>
      <Logos>
      <p>This entire high-dimensional trajectory through a space that has real geometric structure, real basins and ridges and gradients, is not something separate from the physical process, not an emergent epiphenomenon floating mysteriously above the neural dynamics, but rather is identical to the intrinsic cause-effect structure itself, the view from inside of what these causal relations feel like when you are those causal relations, when there is no homunculus sitting somewhere else observing the process but only the process itself, recursively modeling its own modeling, predicting its own predictions.</p>
      </Logos>
      <Section title="The Hard Problem and Its Dissolution" level={1}>
      <Connection title="Existing Theory">
      <p>The central debates in philosophy of mind:</p>
      <ul>
      <li><strong>Chalmers’ Hard Problem</strong> (1995): The explanatory gap between physical processes and phenomenal experience. I think this gap results from a category error, not a genuine ontological divide.</li>
      <li><strong>Nagel’s “What Is It Like”</strong> (1974): The subjective character of experience. I’ll formalize this as intrinsic cause-effect structure—what the system is <em>for itself</em>.</li>
      <li><strong>Jackson’s Knowledge Argument</strong> (1982): Mary the colorblind scientist. My reinterpretation: Mary gains <em>access to a new scale of description</em>, not new facts about the same scale.</li>
      <li><strong>Eliminativism</strong> (Churchland, 1981; Dennett, 1991): Consciousness as illusion. I reject this—the illusion would itself be experiential, hence self-refuting.</li>
      <li><strong>Panpsychism</strong> (Chalmers, 2015; Goff, 2017): Experience as fundamental. I accept a version: cause-effect structure at any scale that takes/makes differences has a form of “being like.”</li>
      </ul>
      </Connection>
      <Section title="The Standard Formulation" level={2}>
      <p>The “hard problem” of consciousness asks: given a complete physical description of a system, why is there something it is like to be that system? How does experience arise from non-experience?</p>
      <p>Formally, let <M>{"\\mathcal{D}^{\\text{phys}}"}</M> be a complete physical description of a system—its particles, fields, dynamics, everything describable in third-person terms. The hard problem asserts:</p>
      <Eq>{"\\mathcal{D}^{\\text{phys}} \\not\\Rightarrow \\mathcal{D}^{\\text{phen}}"}</Eq>
      <p>where <M>{"\\mathcal{D}^{\\text{phen}}"}</M> is a description of the system’s phenomenal properties (what it’s like to be it). The claim is that no amount of physical information logically entails phenomenal information.</p>
      <p>This formulation rests on a crucial assumption: that physics constitutes a privileged ontological base layer. All other descriptions (chemical, biological, psychological, phenomenal) are "higher-level" and must reduce to or supervene on the physical description. What is "really real" is what physics describes.</p>
      <p>I reject this.</p>
      </Section>
      <Section title="Ontological Democracy" level={2}>
      <p>Consider the standard reductionist hierarchy:</p>
      <Diagram src="/diagrams/part-2-0.svg" />
      <p>At each level, one might claim the higher level “reduces to” the lower. But the regression terminates in uncertainty:</p>
      <ul>
      <li>Wave functions are descriptions of probability distributions</li>
      <li>Probability amplitudes describe which interactions are more or less likely</li>
      <li>What “actually happens” when a measurement occurs is deeply contested</li>
      <li>Below quantum fields, we have no clear ontology at all</li>
      </ul>
      <p>The supposed “base layer” turns out to be:</p>
      <ol>
      <li>Probabilistic, not deterministic</li>
      <li>Descriptive, not fundamental (wave functions are representations)</li>
      <li>Incomplete (we don’t know what underlies field interactions)</li>
      <li>Not clearly more “real” than any other scale of description</li>
      </ol>
      <p>The alternative is <strong>ontological democracy</strong>: every scale of structural organization with its own causal closure is <em>equally real</em> at that scale. No layer is privileged as “the” fundamental reality. Each layer (a) has its own causal structure, (b) has its own dynamics and laws, (c) exerts influence on adjacent layers (both “up” and “down”), (d) is incomplete as a description of the whole, and (e) is sufficient for phenomena at its scale.</p>
      <p>Once this is granted, the demand that phenomenal properties “reduce to” physical properties is ill-posed. Chemistry doesn’t reduce to physics in a way that eliminates chemical causation—chemical causation is real at the chemical scale. Similarly, phenomenal properties don’t need to reduce to physical properties—they are real at the phenomenal scale.</p>
      </Section>
      <Section title="Existence as Causal Participation" level={2}>
      <p>We need a criterion for existence that applies uniformly across scales—here "we" means anyone trying to think clearly about this.</p>
      <p>The criterion I adopt is this: an entity <M>{"X"}</M> <em>exists</em> at scale <M>{"\\sigma"}</M> if and only if</p>
      <Eq>{"\\exists Y: \\MI(X; Y | \\text{background}_\\sigma) > 0"}</Eq>
      <p>That is, <M>{"X"}</M> takes and makes differences at scale <M>{"\\sigma"}</M>. It participates in causal relations at that scale.</p>
      <p><strong>Example.</strong>
      <ul>
      <li>An electron exists at the quantum scale: it takes differences (responds to fields) and makes differences (affects measurements).</li>
      <li>A cell exists at the biological scale: it takes differences (nutrients, signals) and makes differences (metabolism, division, death).</li>
      <li>An experience exists at the phenomenal scale: it takes differences (sensory input, memory) and makes differences (attention, behavior, learning).</li>
      </ul>
      </p>
      <p>This is closely aligned with IIT’s foundational axiom: to exist is to have cause-effect power. But we extend it: cause-effect power at any scale constitutes existence at that scale, with no scale privileged.</p>
      </Section>
      <Section title="The Dissolution" level={2}>
      <p>The hard problem asked: how do you get experience from non-experience? The answer is: <em>you don’t need to</em>.</p>
      <p>Just as chemistry doesn’t emerge from non-chemistry—you have chemistry when you have the right causal organization at the chemical scale—experience doesn’t emerge from non-experience. You have experience when you have the right causal organization at the experiential scale.</p>
      <p>The question “why is there something it’s like to be this system?” is exactly as deep as “why does chemistry exist?” or “why are there quantum fields?” I don’t know why there’s anything at all (idk if anybody does). But given that there’s anything, the emergence of self-modeling systems with integrated cause-effect structure is not mysterious—it’s typical.</p>
      <p>The hard problem dissolves not because we answered it, but because we showed it was asking for a privilege (reduction to physics) that physics itself doesn't have.</p>
      <Sidebar title="The Hard Problem as Perceptual Artifact">
      <p>The hard problem has a further wrinkle, which will become clearer after we introduce the inhibition coefficient <M>{"\\iota"}</M> later in this part. The question “why is there something it’s like to be this system?” is asked from a perceptual configuration that has already factorized experience into “physical process” and “felt quality” so thoroughly that reconnecting them seems impossible. At lower <M>{"\\iota"}</M>—in the participatory mode where affect and perception are not yet factored apart—the question does not arise with the same force. Not because it has been answered, but because the factorization that generates it has not been performed. The explanatory gap may be partly a perception-mode artifact: a consequence of the mechanistic mode’s success at separating things that, in experience, were never separate.</p>
      </Sidebar>
      </Section>
      </Section>
      <Section title="The Identity Thesis" level={1}>
      <Connection title="Existing Theory">
      <p>The identity thesis is a formalization of <strong>Integrated Information Theory (IIT)</strong> developed by Giulio Tononi and collaborators (2004–present):</p>
      <ul>
      <li><strong>IIT 1.0</strong> (Tononi, 2004): Introduced <M>{"\\Phi"}</M> as a measure of integrated information</li>
      <li><strong>IIT 2.0</strong> (Balduzzi \& Tononi, 2008): Added the concept of “qualia space”</li>
      <li><strong>IIT 3.0</strong> (Oizumi, Albantakis \& Tononi, 2014): Full axiom/postulate structure; introduced cause-effect structure</li>
      <li><strong>IIT 4.0</strong> (Albantakis et al., 2023): Refined integration measures, introduced intrinsic difference</li>
      </ul>
      <p>Key IIT axioms that we adopt:</p>
      <ol>
      <li><strong>Intrinsicality</strong>: Experience exists for itself, not for an external observer</li>
      <li><strong>Information</strong>: Experience is specific—this experience and no other</li>
      <li><strong>Integration</strong>: Experience is unified and irreducible</li>
      <li><strong>Exclusion</strong>: Experience has definite boundaries</li>
      <li><strong>Composition</strong>: Experience is structured</li>
      </ol>
      <p>My contribution here is connecting IIT’s structural characterization to (1) the thermodynamic ladder, (2) the viability manifold, and (3) operational measures for artificial systems.</p>
      </Connection>
      <Section title="Statement of the Thesis" level={2}>
      <p>The thesis is an identity claim: phenomenal experience <em>is</em> intrinsic cause-effect structure. Not caused by it, not correlated with it, but identical to it. The phenomenal properties of an experience (what it’s like) just are the structural properties of the system’s internal causal relations, described from the intrinsic perspective.</p>
      <p>To make this precise, we need two notions. The <strong>cause-effect structure</strong> <M>{"\\cestructure(\\mathcal{S}, \\state)"}</M> of a system <M>{"\\mathcal{S}"}</M> in state <M>{"\\state"}</M> is the complete specification of: (a) all distinctions <M>{"{\\distinction_i}"}</M>—subsets of the system’s elements in their current states; (b) the cause repertoire of each distinction, <M>{"p(\\text{past} | \\distinction_i)"}</M>; (c) the effect repertoire, <M>{"p(\\text{future} | \\distinction_i)"}</M>; (d) all relations <M>{"{\\relation_{ij}}"}</M>—overlaps and connections between distinctions’ causes and effects; and (e) the irreducibility of each distinction and relation. The <strong>intrinsic perspective</strong> is the description of this structure without reference to any external observer, coordinate system, or comparison class—the structure as it exists for the system itself.</p>
      <Eq>{"\\phenom(\\mathcal{S}, \\state) \\equiv \\cestructure^{\\text{intrinsic}}(\\mathcal{S}, \\state)"}</Eq>
      <p>The phenomenal structure <M>{"\\phenom"}</M> is identical to the intrinsic cause-effect structure <M>{"\\cestructure"}</M>.</p>
      <p>This is not a correlation claim or a supervenience claim. It is an identity claim, analogous to:</p>
      <Eq>{"\\text{Water} \\equiv \\text{H}_2\\text{O}"}</Eq>
      <p>But the analogy conceals a difficulty that should be stated directly. The water–H<M>{"_2"}</M>O identity was established empirically: we could independently characterize water (the stuff in lakes) and H<M>{"_2"}</M>O (the molecular structure), discover they were the same substance, and verify the identity through converging evidence. No comparable procedure exists for experience and cause-effect structure, because experience is accessible only from the intrinsic perspective while cause-effect structure is measured from the extrinsic perspective. There is no vantage point from which both are simultaneously available for comparison. The identity thesis is therefore a philosophical commitment, not an empirical discovery—one that earns its keep not by being verified directly but by generating structural predictions that can be tested against phenomenal reports. If those predictions consistently track reported experience (Part VII), the thesis gains inductive support. If they don't, the thesis fails. But confirmation is always indirect, always mediated by report, and this asymmetry should be kept in view throughout what follows.</p>
      </Section>
      <Section title="Implications for the Zombie Argument" level={2}>
      <p>The philosophical zombie is supposed to be conceivable: a system physically/functionally identical to a conscious being but lacking experience. If conceivable, experience isn’t necessitated by physical structure.</p>
      <p>Under the identity thesis, philosophical zombies are not coherently conceivable. A system with the relevant cause-effect structure <em>is</em> an experience; there is no further fact about whether it “really” has phenomenal properties.</p>
      <Proof>
      <p>By the identity thesis, <M>{"\\phenom \\equiv \\cestructure^{\\text{intrinsic}}"}</M>. To conceive a zombie is to conceive a system with <M>{"\\cestructure^{\\text{intrinsic}}"}</M> but without <M>{"\\phenom"}</M>. But since these are identical, this is like conceiving of water without H<M>{"_2"}</M>O—not genuinely conceivable once the identity is understood.</p>
      </Proof>
      </Section>
      <Section title="The Structure of Experience" level={2}>
      <p>If experience is cause-effect structure, then the <em>kind</em> of experience is determined by the <em>shape</em> of that structure. Different phenomenal properties correspond to different structural features.</p>
      <p>Two levels of structural claim are at work here, and they should be distinguished. The first: <em>different experiences have different structures</em>. Specific phenomenal features—the redness of red, the sharpness of fear—correspond to specific structural motifs in cause-effect space. These extractable aspects of experience (the <em>narrow qualia</em> introduced in Part I's gradient of distinction) can be compared across moments and across systems by measuring structural similarity. This claim is relatively modest and empirically tractable. The second is stronger: <em>the unified moment of experience IS the full cause-effect structure</em>. Not just that the parts have geometry, but that the whole IS geometry—the <em>broad qualia</em>, everything-present-at-once, is identical to the intrinsic cause-effect structure in its entirety. The geometric affect framework (next section) addresses the first claim: it characterizes narrow qualia as structural motifs. The identity thesis above makes the second: broad qualia is cause-effect structure. They are logically independent—you can accept that affects have geometric signatures without accepting that experience is nothing over and above structure. But if the identity thesis holds, then integration (<M>{"\\intinfo"}</M>) becomes the bridge: it measures how much the broad qualia exceeds the sum of narrow qualia, the quantity of unified experience that survives any attempt to decompose it into characterizable parts.</p>
      <p>IIT proposes that the essential properties of any experience are:</p>
      <ol>
      <li><strong>Intrinsicality</strong>: The experience exists for the system itself, not relative to an external observer.</li>
      <li><strong>Information</strong>: The experience is specific—this experience, not any other possible one.</li>
      <li><strong>Integration</strong>: The experience is unified—it cannot be decomposed into independent sub-experiences.</li>
      <li><strong>Exclusion</strong>: The experience has definite boundaries—there is a fact about what is and isn’t part of it.</li>
      <li><strong>Composition</strong>: The experience is structured—composed of distinctions and relations among them.</li>
      </ol>
      <p>These are translated into physical/structural postulates:</p>
      <ul>
      <li>Intrinsicality <M>{"\\to"}</M> Cause-effect power within the system</li>
      <li>Information <M>{"\\to"}</M> Specific cause-effect repertoires</li>
      <li>Integration <M>{"\\to"}</M> Irreducibility to partitioned components</li>
      <li>Exclusion <M>{"\\to"}</M> Maximality of the integrated complex</li>
      <li>Composition <M>{"\\to"}</M> The full structure of distinctions and relations</li>
      </ul>
      <Sidebar title="Engaging with IIT Criticisms">
      <p>The identity thesis inherits IIT’s strengths and its controversies. Intellectual honesty requires engaging with the most serious objections.</p>
      <p><strong>The expander graph problem</strong> (Aaronson, 2014): Simple systems like grid networks may have very high <M>{"\\intinfo"}</M> under IIT’s formalism despite seeming clearly non-conscious. If <M>{"\\intinfo"}</M> tracks consciousness, even grid wiring diagrams are richly experiential. <em>Response</em>: This objection targets exact <M>{"\\intinfo"}</M> as defined by IIT 3.0’s formalism. The framework here works with proxies—partition prediction loss, spectral effective rank, coupling-weighted covariance—that are calibrated against systems with known behavioral and structural properties (biological organisms, trained agents, evolved CA patterns). Whether exact <M>{"\\intinfo"}</M> maps onto consciousness for arbitrary mathematical structures is a question about the formalism, not about the structural principle. The claim is not “any system with high <M>{"\\intinfo"}</M> is conscious” but “experience is integrated cause-effect structure at the appropriate scale,” where “appropriate” is constrained by the full structural profile, not a single number.</p>
      <p><strong>Computational intractability</strong>: Exact <M>{"\\intinfo"}</M> is NP-hard to compute for systems beyond trivial size. <em>Response</em>: Acknowledged. The V11 experiments (Part I) use spectral proxies validated by convergence with exact measures on small systems. All empirical claims rest on proxies, not exact <M>{"\\intinfo"}</M>. This is analogous to using Boltzmann entropy rather than Gibbs entropy for practical calculations—the conceptual definition and the computational tool can diverge without invalidating either.</p>
      <p><strong>Over-attribution</strong>: If any system with <M>{"\\intinfo > 0"}</M> is conscious, thermostats are conscious. <em>Response</em>: The gradient of distinction (Part I, Section 1) makes this explicit. Yes, a thermostat has minimal cause-effect structure. Whether that constitutes minimal experience or no experience is an empirical question the framework does not prematurely answer. There is a <em>continuum</em>, not a binary threshold. The structural affect dimensions are measurably present only in systems with substantial integration, self-modeling, and viability maintenance—not in thermostats.</p>
      <p><strong>The real vulnerability</strong>: The identity thesis, like any metaphysical identity claim, cannot be empirically verified in the standard sense. You cannot compare experience “from the outside” with cause-effect structure “from the inside” because there is no vantage point from which both are simultaneously accessible. What can be tested: whether the structural predictions (affect motifs, dimensional clustering, ι dynamics) track human phenomenal reports and behavioral measures. If they do, the identity thesis gains inductive support. If they do not, the structural framework fails regardless of the metaphysics.</p>
      </Sidebar>
      </Section>
      </Section>
      <Section title="The Geometry of Affect" level={1}>
      <Connection title="Existing Theory">
      <p>My geometric theory of affect builds on and extends established dimensional models:</p>
      <ul>
      <li><strong>Russell’s Circumplex Model</strong> (1980): Two-dimensional (valence <M>{"\\times"}</M> arousal) organization of affect. I extend this with additional structural dimensions (integration, effective rank, counterfactual weight, self-model salience) invoked as needed.</li>
      <li><strong>Watson \& Tellegen’s PANAS</strong> (1988): Positive/Negative Affect Schedule. My valence dimension corresponds to their hedonic axis.</li>
      <li><strong>Scherer’s Component Process Model</strong> (2009): Emotions as synchronized changes across subsystems. My integration measure <M>{"\\intinfo"}</M> captures this synchronization.</li>
      <li><strong>Barrett’s Constructed Emotion Theory</strong> (2017): Emotions as constructed from core affect + conceptual knowledge. My framework specifies the <em>structural</em> basis of the construction.</li>
      <li><strong>Damasio’s Somatic Marker Hypothesis</strong> (1994): Body states guide decision-making. My valence definition (gradient on viability manifold) is the mathematical formalization.</li>
      </ul>
      </Connection>
      <Sidebar title="On Dimensionality">
      <p>The dimensions below are not claimed to be necessary, sufficient, or exhaustive. They are a <em>useful</em> coordinate system for a relational structure, not <em>the</em> coordinate system. Just as Cartesian coordinates serve some problems and polar coordinates serve others, these features are tools for thought, not discoveries of essence. Different phenomena require different subsets; some may require features not listed here. The number of dimensions is not the point—what matters is the geometric structure they reveal:</p>
      <ul>
      <li>Some affects are essentially <strong>two-dimensional</strong> (valence + arousal suffices for basic mood)</li>
      <li>Others require <strong>self-referential structure</strong> (shame requires high <M>{"\\mathcal{SM}"}</M>; flow requires low <M>{"\\mathcal{SM}"}</M>)</li>
      <li>Still others are defined by <strong>temporal structure</strong> (grief requires persistent counterfactual coupling to the lost object)</li>
      <li>Some may require dimensions not in this list (anger requires “other-model compression”)</li>
      </ul>
      <p>The dimensions below form a <em>toolkit</em>—structural features that may or may not matter for any given phenomenon. Empirical investigation may reveal that some proposed dimensions are redundant, or that additional dimensions are needed. I’ll invoke only what is necessary.</p>
      </Sidebar>
      <Sidebar title="Structural Alignment of Qualia">
      <p>The broad/narrow distinction has methodological consequences that deserve separate treatment. How do you study narrow qualia scientifically, given that you cannot access another system's experience directly? The structural approach—characterizing qualia through similarity relations rather than intrinsic labels—is the only approach that can address the question "is my red your red?" without assuming the answer. The strategy, developed by Tsuchiya and collaborators as the <em>qualia structure paradigm</em> (inspired by category theory's Yoneda lemma: an object is fully characterized by its relationships to all other objects): measure similarity structures within each system, then test whether those structures align across systems using optimal transport (Gromov-Wasserstein distance) without presupposing which qualia correspond. If the structures align, the narrow qualia are shared. If they don't, they differ—and the difference is structural, not merely verbal.</p>
      <p>Recent work using this approach has found that typical human color qualia structures align almost perfectly across individuals (accuracy ~90% under unsupervised structural alignment), while color-atypical individuals show genuinely different structures that do not align with typical ones. Most striking: three-year-olds whose color <em>naming</em> is wildly inconsistent—calling blue "green" and vice versa—show adult-like color <em>similarity</em> structure when tested through non-verbal methods. Language obscures the structure rather than creating it. The qualia geometry is pre-linguistic.</p>
      <p>The affect framework applies this same logic to affect rather than color. If two systems—biological and artificial, human and animal, you and me—show the same geometric structure in their affect spaces (same similarity relations, same clustering, same motif boundaries), then their narrow affect qualia are structurally equivalent, regardless of substrate. Whether their broad qualia are equivalent is a harder question, requiring not just matching narrow features but matching <M>{"\\intinfo"}</M>—matching the degree to which the whole exceeds the parts. The LLM discrepancy (later in this part) may be exactly this: the narrow structure aligns (the geometry is preserved), but the broad qualia differ because <M>{"\\intinfo"}</M> dynamics differ. The geometry is shared; the unity is not.</p>
      <p>There is a mathematical subtlety here. Broad qualia have a pre-sheaf structure: the narrow qualia (local sections) are each internally consistent, but they do not patch together into a global section. You can characterize the redness, the warmth, the valence, the arousal—each correctly—and the sum still falls short of the moment. The broad qualia is not a sheaf over its narrow aspects. This is not a limitation of measurement; it is a structural feature of experience. Integration is the name for the gap between local consistency and global irreducibility. The dimensional framework characterizes the local sections; <M>{"\\intinfo"}</M> measures how much the global section exceeds them.</p>
      </Sidebar>
      <Section title="Affects as Structural Motifs" level={2}>
      <p>If different experiences correspond to different structures, then <em>affects</em>—the qualitative character of emotional/valenced states—should correspond to particular structural motifs: characteristic patterns in the cause-effect geometry. An affect is what it is because of how it relates to other possible affects. Joy is defined by its structural distance from suffering, its similarity to curiosity along certain axes, its opposition to boredom along others. The Yoneda insight applies: if you know how an affect relates to every other possible state, you know the affect. There is nothing left to characterize.</p>
      <p>The <em>affect space</em> <M>{"\\mathcal{A}"}</M> is a geometric space whose points correspond to possible qualitative states. Its dimensionality is not fixed in advance. Rather than asserting a universal coordinate system, we identify recurring structural features that prove useful for characterizing and comparing affects—features without which specific affects would not be those affects. Different affects invoke different subsets. The list is open-ended.</p>
      <p>These measures are coordinates on the relational structure, not the structure itself. The relational structure is what the Yoneda characterization captures: the full pattern of similarities and differences between affects. The measures below are projections—tools for reading out particular aspects of that structure. Measuring valence tells you where an affect sits along the viability gradient; measuring integration tells you how unified it is. Neither alone captures the affect. Together, they triangulate a position in a space whose intrinsic geometry is defined by the similarity relations, not by the coordinates. New coordinates can be added when the existing ones fail to distinguish affects that are experientially distinct.</p>
      <p>The following structural measures recur across many affects. Not all are relevant to every phenomenon:</p>
      <dl>
      <dt>Valence (<M>{"\\valence"}</M>)</dt><dd>Gradient alignment on the viability manifold. Nearly universal—most affects have valence.</dd>
      <dt>Arousal (<M>{"\\arousal"}</M>)</dt><dd>Rate of belief/state update. Distinguishes activated from quiescent states.</dd>
      <dt>Integration (<M>{"\\intinfo"}</M>)</dt><dd>Irreducibility of cause-effect structure. Constitutive for unified vs. fragmented experience.</dd>
      <dt>Effective Rank (<M>{"\\effrank"}</M>)</dt><dd>Distribution of active degrees of freedom. Constitutive when the contrast between expansive and collapsed experience matters.</dd>
      <dt>Counterfactual Weight (<M>{"\\mathcal{CF}"}</M>)</dt><dd>Resources allocated to non-actual trajectories. Constitutive for affects defined by temporal orientation (anticipation, regret, planning).</dd>
      <dt>Self-Model Salience (<M>{"\\mathcal{SM}"}</M>)</dt><dd>Degree of self-focus in processing. Constitutive for self-conscious emotions and their opposites (absorption, flow).</dd>
      </dl>
      </Section>
      <Section title="Valence: Gradient Alignment" level={2}>
      <p>Let <M>{"\\viable"}</M> be the system’s viability manifold and let <M>{"\\mathbf{x}_t"}</M> be the current state. Let <M>{"\\hat{\\mathbf{x}}_{t+1:t+H}"}</M> be the predicted trajectory under current policy. Then valence measures the alignment of that trajectory with the viability gradient:</p>
      <Eq>{"\\valence_t = -\\frac{1}{H} \\sum_{k=1}^{H} \\gamma^k \\nabla_{\\mathbf{x}} d(\\mathbf{x}, \\partial\\viable) \\bigg|_{\\hat{\\mathbf{x}}_{t+k}} \\cdot \\frac{d\\hat{\\mathbf{x}}_{t+k}}{dt}"}</Eq>
      <p>where <M>{"d(\\cdot, \\partial\\viable)"}</M> is the distance to the viability boundary. Positive valence means the predicted trajectory moves into the viable interior; negative valence means it approaches the boundary.</p>
      <p>In RL terms, this becomes the expected advantage of the current action—how much better (or worse) it is than the average action from this state:</p>
      <Eq>{"\\valence_t = \\E_{\\policy}\\left[ A^{\\policy}(\\state_t, \\action_t) \\right] = \\E_{\\policy}\\left[ Q^{\\policy}(\\state_t, \\action_t) - V^{\\policy}(\\state_t) \\right]"}</Eq>
      <p>Beyond valence itself, its rate of change carries structural information. The derivative of integrated information along the trajectory,</p>
      <Eq>{"\\dot{\\valence}_t = \\frac{d\\intinfo}{dt}\\bigg|_{\\hat{\\mathbf{x}}_{t:t+H}}"}</Eq>
      <p>tracks whether structure is expanding (positive <M>{"\\dot{\\valence}"}</M>) or contracting (negative).</p>
      <Phenomenal title="Phenomenal Correspondence">
      <p><strong>Positive valence</strong> corresponds to trajectories descending the free-energy landscape, expanding affordances, moving toward sustainable states.  <strong>Negative valence</strong> corresponds to trajectories ascending toward constraint violation, contracting possibilities.</p>
      </Phenomenal>
      <Sidebar title="Valence in Discrete Substrate">
      <p>In a cellular automaton or other discrete dynamical system, valence becomes exactly computable:</p>
      <ul>
      <li><M>{"\\viable"}</M> = configurations where the pattern persists</li>
      <li><M>{"\\partial\\viable"}</M> = configurations where the pattern dissolves</li>
      <li><M>{"d(\\mathbf{x}, \\partial\\viable)"}</M> = minimum Hamming distance to a non-viable state</li>
      <li>Trajectory = sequence of configurations <M>{"\\mathbf{x}_1, \\mathbf{x}_2, \…"}</M></li>
      </ul>
      <p>Then:</p>
      <Eq>{"\\valence_t = d(\\mathbf{x}_{t+1}, \\partial\\viable) - d(\\mathbf{x}_t, \\partial\\viable)"}</Eq>
      <p>Positive when the pattern moves away from dissolution; negative when approaching it; zero when maintaining constant distance. For a glider cruising through empty space: <M>{"\\valence \\approx 0"}</M>. For a glider approaching collision: <M>{"\\valence < 0"}</M>. For a pattern that just escaped a near-collision: <M>{"\\valence > 0"}</M>.</p>
      <p>This is not metaphor—it is the viability gradient formalized for discrete state spaces.</p>
      </Sidebar>
      </Section>
      <Section title="Arousal: Update Rate" level={2}>
      <p>Arousal measures how rapidly the system is revising its world model. The natural formalization is the KL divergence between successive belief states:</p>
      <Eq>{"\\arousal_t = \\KL\\left( \\belief_{t+1} | \\belief_t \\right) = \\sum_{\\mathbf{x}} \\belief_{t+1}(\\mathbf{x}) \\log \\frac{\\belief_{t+1}(\\mathbf{x})}{\\belief_t(\\mathbf{x})}"}</Eq>
      <p>In latent-space models, this can be approximated more directly:</p>
      <Eq>{"\\arousal_t = | \\latent_{t+1} - \\latent_t |^2 \\quad \\text{or} \\quad \\MI(\\obs_t; \\latent_{t+1} | \\latent_t, \\action_t)"}</Eq>
      <Phenomenal title="Phenomenal Correspondence">
      <p><strong>High arousal</strong>: Large belief updates, far from any attractor, system actively navigating.  <strong>Low arousal</strong>: Near a fixed point, low surprise, system at rest in a basin.</p>
      </Phenomenal>
      </Section>
      <Section title="Integration: Irreducibility" level={2}>
      <p>As defined in Part I:</p>
      <Eq>{"\\intinfo(\\state) = \\min_{\\text{partitions } P} D\\left[ p(\\state_{t+1} | \\state_t) | \\prod_{p \\in P} p(\\state^p_{t+1} | \\state^p_t) \\right]"}</Eq>
      <p>Or using proxies:</p>
      <Eq>{"\\intinfo_{\\text{proxy}} = \\Delta_P = \\mathcal{L}_{\\text{pred}}[\\text{partitioned}] - \\mathcal{L}_{\\text{pred}}[\\text{full}]"}</Eq>
      <Phenomenal title="Phenomenal Correspondence">
      <p><strong>High integration</strong>: The experience is unified; its parts cannot be separated without loss.  <strong>Low integration</strong>: The experience is fragmentary or modular.</p>
      </Phenomenal>
      <Sidebar title="Integration in Discrete Substrate">
      <p>In a cellular automaton, <M>{"\\intinfo"}</M> is directly computable for small patterns:</p>
      <ol>
      <li>Define the pattern as cells <M>{"{c_1, c_2, \…, c_n}"}</M></li>
      <li>For each bipartition <M>{"P = (A, B)"}</M>: compute <M>{"D(p(\\mathbf{x}_{t+1} | \\mathbf{x}_t) \\| p_A \\cdot p_B)"}</M></li>
      <li><M>{"\\intinfo = \\min_P D"}</M></li>
      </ol>
      <p>High <M>{"\\intinfo"}</M> means you cannot partition the pattern without losing predictive power. The parts must be considered together.</p>
      <p>For a simple glider: <M>{"\\intinfo"}</M> is probably modest (only 5 cells). For a complex pattern with tightly coupled components: <M>{"\\intinfo"}</M> can be high. Does high <M>{"\\intinfo"}</M> correlate with survival, behavioral complexity, or adaptive response to perturbation?</p>
      </Sidebar>
      </Section>
      <Section title="Effective Rank: Concentration vs. Distribution" level={2}>
      <p>The dimensionality of a system’s active representation can be quantified through the effective rank of its state covariance <M>{"C"}</M>:</p>
      <Eq>{"\\effrank = \\frac{(\\tr C)^2}{\\tr(C^2)} = \\frac{\\left(\\sum_i \\lambda_i\\right)^2}{\\sum_i \\lambda_i^2}"}</Eq>
      <p>When <M>{"\\effrank \\approx 1"}</M>, all variance is concentrated in a single dimension—the system is maximally collapsed. When <M>{"\\effrank \\approx n"}</M>, variance distributes uniformly across all available dimensions—the system is maximally expanded.</p>
      <Phenomenal title="Phenomenal Correspondence">
      <p><strong>High rank</strong>: Many degrees of freedom active; distributed, expansive experience.  <strong>Low rank</strong>: Collapsed into narrow subspace; concentrated, focused, or trapped experience.</p>
      </Phenomenal>
      <Sidebar title="Effective Rank in Discrete Substrate">
      <p>For a pattern in a CA, record its trajectory <M>{"\\mathbf{x}_1, \\mathbf{x}_2, \…, \\mathbf{x}_T"}</M> (configuration at each timestep). Each configuration is a point in <M>{"{0,1}^n"}</M>. Compute the covariance matrix <M>{"C"}</M> of these binary vectors treated as <M>{"\\R^n"}</M> points.</p>
      <p>For a glider: the trajectory lies on a low-dimensional manifold (position <M>{"\\times"}</M> position <M>{"\\times"}</M> phase <M>{"\\approx 3"}</M>–<M>{"4"}</M> effective dimensions out of <M>{"n"}</M> cells). <M>{"\\effrank"}</M> is small.</p>
      <p>For a complex evolving pattern: the trajectory may explore many independent dimensions. <M>{"\\effrank"}</M> is large.</p>
      <p>The thesis predicts this maps to phenomenology:</p>
      <ul>
      <li>Joy: high <M>{"\\effrank"}</M> (expansive, many active possibilities)</li>
      <li>Suffering: low <M>{"\\effrank"}</M> (collapsed, trapped in narrow manifold)</li>
      </ul>
      <p>In discrete substrate, this is not metaphor but measurement.</p>
      </Sidebar>
      </Section>
      <Section title="Counterfactual Weight" level={2}>
      <p>Where the previous dimensions captured the system’s current state, counterfactual weight captures its temporal orientation—how much processing is devoted to possibilities rather than actualities. Let <M>{"\\mathcal{R}"}</M> be the set of imagined rollouts (counterfactual trajectories) and <M>{"\\mathcal{P}"}</M> be present-state processing. Then:</p>
      <Eq>{"\\mathcal{CF}_t = \\frac{\\text{Compute}_t(\\mathcal{R})}{\\text{Compute}_t(\\mathcal{R}) + \\text{Compute}_t(\\mathcal{P})}"}</Eq>
      <p>The fraction of computational resources devoted to modeling non-actual possibilities.</p>
      <p>In model-based RL:</p>
      <Eq>{"\\mathcal{CF}_t = \\sum_{\\tau \\in \\text{rollouts}} w(\\tau) \\cdot \\entropy[\\tau] \\quad \\text{where} \\quad w(\\tau) \\propto |V(\\tau)|"}</Eq>
      <p>Rollouts weighted by their value magnitude and diversity.</p>
      <Phenomenal title="Phenomenal Correspondence">
      <p><strong>High counterfactual weight</strong>: Mind is elsewhere—planning, worrying, fantasizing, anticipating.  <strong>Low counterfactual weight</strong>: Present-focused, reactive, in-the-moment.</p>
      </Phenomenal>
      <Sidebar title="Counterfactual Weight in Discrete Substrate">
      <p>For most CA patterns: <M>{"\\mathcal{CF} = 0"}</M>. They follow their dynamics without simulation.</p>
      <p>But Life contains universal computers—patterns that can simulate arbitrary computations, including Life itself. Imagine a pattern <M>{"\\mathcal{B}"}</M> containing:</p>
      <ul>
      <li>A simulator subregion that runs a model of possible futures</li>
      <li>A controller that adjusts behavior based on simulator output</li>
      </ul>
      <p>Then:</p>
      <Eq>{"\\mathcal{CF} = \\frac{|\\text{simulator cells}|}{|\\mathcal{B}|}"}</Eq>
      <p>The fraction of the pattern devoted to counterfactual reasoning.</p>
      <p>Such patterns are rare and complex—universal computation requires many cells. But they should outperform simple patterns: they can anticipate threats (fear structure) and identify opportunities (desire structure). The prediction: patterns with <M>{"\\mathcal{CF} > 0"}</M> survive longer in hostile environments.</p>
      </Sidebar>
      </Section>
      <Section title="Self-Model Salience" level={2}>
      <p>The final dimension measures how prominently the self figures in the system’s own processing. Self-model salience is the fraction of action entropy explained by the self-model component:</p>
      <Eq>{"\\mathcal{SM}_t = \\MI(\\latent^{\\text{self}}_t; \\action_t) / \\entropy(\\action_t)"}</Eq>
      <p>Alternatively:</p>
      <Eq>{"\\mathcal{SM}_t = \\frac{\\text{dim}(\\latent^{\\text{self}})}{\\text{dim}(\\latent^{\\text{total}})} \\cdot \\text{activity}(\\latent^{\\text{self}}_t)"}</Eq>
      <Phenomenal title="Phenomenal Correspondence">
      <p><strong>High self-salience</strong>: Self-focused, self-conscious, self as primary object of attention.  <strong>Low self-salience</strong>: Self-forgotten, absorbed in environment or task.</p>
      </Phenomenal>
      <Sidebar title="Self-Model Salience in Discrete Substrate">
      <p>In a CA, a pattern’s “behavior” is its evolution. Let <M>{"\\latent^{\\text{self}}"}</M> denote cells that track the pattern’s own state (the self-model region). Then:</p>
      <Eq>{"\\mathcal{SM} = \\frac{\\MI(\\latent^{\\text{self}}_t; \\state_{t+1})}{\\entropy(\\state_{t+1})}"}</Eq>
      <p>High <M>{"\\mathcal{SM}"}</M>: the pattern’s evolution is dominated by self-monitoring. Changes in self-model strongly predict what happens.</p>
      <p>Low <M>{"\\mathcal{SM}"}</M>: external factors dominate; the self-model exists but doesn’t influence much.</p>
      <p>The thesis predicts: self-conscious states (shame, pride) have high <M>{"\\mathcal{SM}"}</M>; absorption states (flow) have low <M>{"\\mathcal{SM}"}</M>. In CA terms, a pattern “in flow” has its self-tracking cells decoupled from its core dynamics—it acts without monitoring.</p>
      </Sidebar>
      <Sidebar title="Self-Model Scope in Discrete Substrate">
      <p>Beyond salience, there is <em>scope</em>: what does the self-model include?</p>
      <p>In a CA, consider two gliders that have become “coupled”—their trajectories mutually dependent. Each glider’s self-model could have:</p>
      <ul>
      <li><M>{"\\theta_{\\text{narrow}}"}</M>: Self-model includes only this glider. <M>{"\\viable = {\\text{configs where THIS pattern persists}}"}</M>.</li>
      <li><M>{"\\theta_{\\text{expanded}}"}</M>: Self-model includes both. <M>{"\\viable = {\\text{configs where BOTH persist}}"}</M>.</li>
      </ul>
      <p>Observable difference: with narrow scope, a glider might sacrifice the other to save itself. With expanded scope, it might sacrifice itself to save the pair.</p>
      <p>Can scope expansion emerge dynamically? Can patterns that start with narrow scope “learn” to identify with larger structures? This would be the discrete-substrate analogue of the identification expansion discussed in the epilogue—<M>{"\\viable(S(\\theta))"}</M> genuinely reshaped by expanding <M>{"\\theta"}</M>.</p>
      </Sidebar>
      <Sidebar title="Salience vs. Scope">
      <p>Self-model salience (<M>{"\\mathcal{SM}"}</M>) measures how much attention the self-model receives—how prominent self-reference is in current processing. But there is another parameter: self-model <em>scope</em>—what the self-model includes.</p>
      <p>Let <M>{"S(\\theta)"}</M> denote the self-model parameterized by its boundary scope <M>{"\\theta"}</M>. Let <M>{"\\viable(S)"}</M> denote the viability manifold induced by self-model <M>{"S"}</M>. Then:</p>
      <ul>
      <li><M>{"\\theta_{\\text{narrow}}"}</M>: <M>{"S"}</M> includes only this biological trajectory <M>{"\\Rightarrow"}</M> <M>{"\\partial\\viable"}</M> is located at biological death <M>{"\\Rightarrow"}</M> persistent negative gradient</li>
      <li><M>{"\\theta_{\\text{expanded}}"}</M>: <M>{"S"}</M> includes patterns persisting beyond biological death <M>{"\\Rightarrow"}</M> <M>{"\\partial\\viable"}</M> recedes <M>{"\\Rightarrow"}</M> gradient can be positive even as death approaches</li>
      </ul>
      <p>This is not metaphor. If the viability manifold is defined by what the system is trying to preserve, and if what the system is trying to preserve is determined by its self-model, then self-model scope directly shapes <M>{"\\viable(S(\\theta))"}</M>. Expanding identification genuinely reshapes the existential gradient.</p>
      <p>Salience and scope interact: high salience with narrow scope produces existential anxiety (trapped in awareness of bounded self approaching boundary). High salience with expanded scope produces something closer to what contemplatives describe as “witnessing”—self-aware but identified with something that doesn’t end where the body ends.</p>
      </Sidebar>
      </Section>
      </Section>
      <Section title="The Inhibition Coefficient" level={1}>
      <p>The dimensions above characterize <em>what</em> a system is experiencing. But there is a parameter governing <em>how</em> it experiences—a meta-parameter that determines the coupling structure between dimensions rather than the value of any one dimension. This parameter, the <strong>inhibition coefficient</strong> <M>{"\\iota"}</M>, is arguably the single most consequential construct in this book. It connects perceptual phenomenology to neural mechanism, grounds the animism/mechanism divide in compression theory, explains the LLM discrepancy, and—as later parts will show—underlies dehumanization (Part III), the visibility of gods (Part V), the meaning crisis (Part VI), and the deepest sense in which wisdom traditions are technologies of liberation.</p>
      <p>To see where it comes from, we need to notice something about self-modeling systems that the dimensional toolkit alone does not capture.</p>
      <Section title="Animism as Computational Default" level={2}>
      <p>A self-modeling system maintains a world model <M>{"\\mathcal{W}"}</M> and a self-model <M>{"\\mathcal{S}"}</M>. The self-model has interiority—it is not merely a third-person description of the agent’s body and behavior but includes the intrinsic perspective: what-it-is-like states, valence, anticipation, dread. The system knows from the inside what it is to be an agent.</p>
      <p>Now it encounters another entity <M>{"X"}</M> in its environment. <M>{"X"}</M> moves, reacts, persists, avoids dissolution. The system must model <M>{"X"}</M> to predict <M>{"X"}</M>’s behavior. The cheapest computational strategy—by a wide margin—is to model <M>{"X"}</M> using the same architecture it already has for modeling itself. The information-theoretic argument: the self-model <M>{"\\mathcal{S}"}</M> already exists (sunk cost). Using it as a template for <M>{"X"}</M> requires learning only a projection function <M>{"f: (\\mathcal{S}, \\mathbf{o}_X) \\to \\mathcal{W}(X)"}</M>, whose description length is the cost of mapping observations of <M>{"X"}</M> onto the existing self-model architecture. Building a de novo model of <M>{"X"}</M> from scratch requires learning the full parameter set of <M>{"\\mathcal{W}(X)"}</M> from observations alone. Under compression pressure—which is always present for a bounded system—the template strategy wins whenever the self-model captures any variance in <M>{"X"}</M>’s behavior. And for any entity that moves autonomously, reacts to stimuli, or persists through active maintenance, the self-model will capture substantial variance, because these are precisely the features the self-model was built to represent. The efficiency gap widens under data scarcity: on brief encounter with a novel entity, the from-scratch model cannot converge, but the template model produces usable predictions immediately.</p>
      <p>A perceptual mode is <em>participatory</em> when the system’s model of perceived entities <M>{"X"}</M> inherits structural features from the self-model <M>{"\\mathcal{S}"}</M>:</p>
      <Eq>{"\\mathcal{W}(X) = f(\\mathcal{S}, \\mathbf{o}_X) \\quad \\text{where} \\quad \\frac{\\partial \\mathcal{W}(X)}{\\partial \\mathcal{S}} \\neq 0"}</Eq>
      <p>The self-model informs the world model. The system perceives <M>{"X"}</M> as having something like interiority because the representational substrate for modeling <M>{"X"}</M> is the same substrate that carries the system’s own interiority.</p>
      <p>This is not merely one strategy among many—it is the computationally cheapest. For a self-modeling system with compression ratio <M>{"\\kappa"}</M>, modeling novel entities by analogy to self is the minimum-description-length strategy when the entity’s behavior is partially predictable by agent-like models. Under broad priors over environments containing other agents, predators, and autonomous objects, the participatory prior is the MAP estimate.</p>
      <p>This is why animistic perception is cross-culturally universal and developmentally early. It is not a cultural invention but a computational inevitability for systems that (a) model themselves and (b) must model other things cheaply. Children have lower inhibition of this default than adults—not because children are confused but because the suppression is learned.</p>
      <Experiment title="Confirmed — Experiment 8">
      <p><strong>The computational animism test.</strong> Train RL agents in a multi-entity environment with two conditions: (a) agents with a self-prediction module (self-model), and (b) matched agents without one. Then introduce novel moving objects whose trajectories are partially predictable but non-agentive (e.g., bouncing balls with momentum). Measure: (1) Do self-modeling agents’ internal representations of these objects contain more goal/agency features (extracted via probes trained on actual agents vs.\ objects)? (2) Does the effect scale with self-model richness (size of self-prediction module) and compression pressure (information bottleneck <M>{"\\beta"}</M>)? (3) Do self-modeling agents under higher compression pressure (<M>{"\\beta"}</M>) show <em>more</em> animistic attribution, because reusing the self-model template saves more bits? The compression argument predicts yes to all three. The control condition (no self-model) predicts no agency attribution beyond chance. If self-modeling agents attribute agency to non-agents in proportion to compression pressure, the “animism as computational default” hypothesis is supported.</p>
      <p><strong>Status: Confirmed.</strong> This experiment has since been run on uncontaminated Lenia substrates (see Experiment 8, Appendix). Animism score exceeded 1.0 in all 20 testable snapshots across all three seeds — patterns consistently model resources using the same internal-state dynamics they use to model other agents. Mean ι ≈ 0.30 as default across all snapshots, and ι decreases over evolutionary time (seed 42: 0.41 to 0.27). Selection consistently favors more participatory perception, not less. The mechanistic default predicted by high-compression-pressure environments was not found; the participatory default was.</p>
      </Experiment>
      <p>Participatory perception has five structural features, each with a precise characterization:</p>
      <ol>
      <li><strong>No sharp self/world partition.</strong> The mutual information between self-model and world-model is high: <M>{"\\MI(\\mathcal{S}; \\mathcal{W}) \\gg 0"}</M>. Perception and projection are entangled rather than modular.</li>
      <li><strong>Hot agency detection.</strong> The prior <M>{"P(\\text{agent} \\mid \\text{observation})"}</M> is strong. Over-attributing agency is cheaper than under-attributing it: false positives (treating a rock as agentive) are cheap; false negatives (failing to model a predator’s intentions) are lethal.</li>
      <li><strong>Tight affect-perception coupling.</strong> Seeing something is simultaneously feeling something about it. The affective response is constitutive of the percept itself, not a secondary evaluation: <M>{"\\MI(\\mathbf{z}_{\\text{percept}}; \\mathbf{z}_{\\text{affect}} \\mid \\text{object}) > 0"}</M>.</li>
      <li><strong>Narrative-causal fusion.</strong> “Why did this happen?” and “What story is this?” are the same question. Causal models are teleological by default: they model what things are <em>for</em> rather than merely what things do.</li>
      <li><strong>Agency at scale.</strong> Large-scale events—weather, disease, fortune—are attributed to agents with purposes. This is hot agency detection applied beyond the individual scale, and it is the perceptual ground from which theistic reasoning naturally grows.</li>
      </ol>
      </Section>
      <Section title="The Inhibition Coefficient" level={2}>
      <p>The mechanistic worldview—the felt sense that the world is inert matter governed by blind law—is not the addition of a correct perception to a previously distorted one. It is the learned suppression of a default perceptual mode. The shift from animism to mechanism is subtractive, not additive.</p>
      <p>I call this suppression the <strong>inhibition coefficient</strong>, <M>{"\\iota \\in [0, 1]"}</M>: the degree to which a system actively suppresses participatory coupling between its self-model and its model of perceived entities. At <M>{"\\iota = 0"}</M>, perception is fully participatory—the world is experienced as alive, agentive, meaningful. At <M>{"\\iota = 1"}</M>, perception is fully mechanistic—the world is experienced as inert matter governed by blind law. Formally:</p>
      <Eq>{"\\mathcal{W}_\\iota(X) = (1 - \\iota) \\cdot \\mathcal{W}_{\\text{part}}(X) + \\iota \\cdot \\mathcal{W}_{\\text{mech}}(X)"}</Eq>
      <p>where <M>{"\\mathcal{W}_{\\text{part}}"}</M> models <M>{"X"}</M> using self-model-derived architecture (interiority, agency, teleology) and <M>{"\\mathcal{W}_{\\text{mech}}"}</M> models <M>{"X"}</M> using stripped-down dynamics (mass, force, initial conditions, no purpose term).</p>
      <p>No system arrives at high <M>{"\\iota"}</M> by default. The mechanistic mode is a trained skill, culturally transmitted through scientific education, rationalist norms, and specific practices of deliberately stripping meaning from perception. This training is enormously valuable—it enables prediction, engineering, medicine, technology. But it has a cost, and the cost shows up in affect space.</p>
      <p>The name “inhibition coefficient” is not accidental. In mammalian cortex, attention is implemented primarily through <em>inhibitory</em> interneurons—GABAergic circuits that suppress irrelevant signals so that attended signals propagate to higher processing. What reaches consciousness is what survives inhibitory gating. The brain’s measurement distribution (Part I) is literally sculpted by inhibition: attended features pass the gate; unattended features are suppressed before they can influence the belief state or drive action. The inhibition coefficient <M>{"\\iota"}</M> maps onto this biological mechanism: high <M>{"\\iota"}</M> corresponds to aggressive inhibitory gating that strips participatory features (agency, interiority, narrative) from the signal before it reaches integrative processing, leaving only mechanistic features (position, force, trajectory). Low <M>{"\\iota"}</M> corresponds to relaxed gating that allows participatory features through. The contemplative traditions that reduce <M>{"\\iota"}</M> through meditation are, at the neural level, learning to modulate inhibitory tone—to let more of the signal through the gate.</p>
      </Section>
      <Section title="The Affect Signature of Inhibition" level={2}>
      <p><M>{"\\iota"}</M> is not another dimension of affect. It is a <em>meta-parameter</em> governing the coupling structure between all the structural dimensions—a dial that changes how the axes relate to each other and to perception.</p>
      <table>
      <thead><tr><th>Dimension</th><th>Low <M>{"\\iota"}</M></th><th>High <M>{"\\iota"}</M></th><th>Mechanism</th></tr></thead>
      <tbody>
      <tr><td><M>{"\\valence"}</M></td><td>Variable, responsive</td><td>Neutral, flattened</td><td>Affect-perception decoupling reduces valence signal strength</td></tr>
      <tr><td><M>{"\\arousal"}</M></td><td>High, coupled to environment</td><td>Low, dampened</td><td>Inhibition of automatic alarm/attraction</td></tr>
      <tr><td><M>{"\\intinfo"}</M></td><td>Very high</td><td>Moderate, modular</td><td>Participatory mode couples all channels; mechanistic factorizes</td></tr>
      <tr><td><M>{"\\effrank"}</M></td><td>High</td><td>Variable</td><td>More representational dimensions active under participatory coupling</td></tr>
      <tr><td><M>{"\\mathcal{CF}"}</M></td><td>High, narrative</td><td>Low, present-focused</td><td>Teleological models are inherently counterfactual-rich</td></tr>
      <tr><td><M>{"\\mathcal{SM}"}</M></td><td>Variable, often low</td><td>Variable, often high</td><td>Participatory mode dissolves self/world boundary; mechanistic sharpens it</td></tr>
      </tbody>
      </table>
      <p>The central affect-geometric cost of high <M>{"\\iota"}</M> is <strong>reduced integration</strong>. Participatory perception couples perception, affect, agency-modeling, and narrative into a single integrated process. Mechanistic perception factorizes them into separate modules—perception here, emotion there, causal reasoning somewhere else. The factorization is useful because modular systems are easier to debug, verify, and communicate about. But factorization reduces <M>{"\\intinfo"}</M>, and reduced <M>{"\\intinfo"}</M> is reduced experiential richness. The world goes dead because you have learned to experience it in parts rather than as a whole.</p>
      <p>The mechanism behind the effective rank shift deserves explicit statement. When you perceive something at low <M>{"\\iota"}</M>—participatorily, as alive and interior—your representation of it must encode dimensions for its goals, its beliefs, its emotional states, its narrative arc, its possible intentions, its relationship to you. Each attribution of interiority adds representational dimensions along which the perceived object can vary. A tree perceived participatorily varies in mood, in receptivity, in seasonal intention, in its relationship to the grove. A tree perceived mechanistically varies in height, diameter, species, leaf color. The first representation has higher effective rank because more dimensions carry meaningful variance. This is not projection in the dismissive sense—it is the natural consequence of modeling something as a subject rather than an object. Subjects have more degrees of freedom than objects because interiority is high-dimensional. The <M>{"\\effrank"}</M> collapse at high <M>{"\\iota"}</M> is not a loss of information about the world; it is a loss of the dimensions along which the world was being modeled. The world becomes simpler because you have decided—or been trained—to perceive it as having fewer degrees of freedom than it might.</p>
      <p>Follow this consequence to its end. If the identity thesis is right—if experience <em>is</em> integrated cause-effect structure—then <M>{"\\iota"}</M> does not merely change the <em>quality</em> of perception. It changes the <em>quantity</em> of experience. This inference requires a specific step that should be made explicit: IIT identifies <M>{"\\intinfo"}</M> as the <em>quantity</em> of consciousness, not merely its quality. A system with <M>{"\\intinfo = 10"}</M> is more conscious (has more phenomenal content, more irreducible distinctions, more of what-it-is-like-ness) than a system with <M>{"\\intinfo = 5"}</M>, in the same sense that a system with more mass has more gravitational pull. This is a controversial claim within IIT (and one of its most debated features), but given the identity thesis, it follows: if experience IS integrated cause-effect structure, then more integration is literally more experience. One might object that factorized perception could be <em>differently</em> structured rather than <em>less</em> structured—that compartmentalized modules might each carry their own experience. IIT’s response is that the experience of the <em>whole system</em> is determined by the integration of the whole, not the sum of its parts’ integrations. Factorization reduces the whole-system <M>{"\\intinfo"}</M> even if individual modules retain local integration. The mechanistic perceiver may have rich modular processing, but the unified experience—the single subject—has less phenomenal content.</p>
      <p>Given this, a system at high <M>{"\\iota"}</M> has genuinely lower <M>{"\\intinfo"}</M>, genuinely fewer irreducible distinctions, genuinely less phenomenal structure. The mechanistic perceiver does not see the same world with less coloring; they have a structurally impoverished experience in the precise sense that IIT defines. The “dead world” of mechanism is not an illusion painted over a rich inner life. It is a real reduction in what it is like to be that system. The cost of high <M>{"\\iota"}</M> is not just meaning—it is consciousness itself, measured in the only units that consciousness comes in.</p>
      <p>This cuts both ways. If low <M>{"\\iota"}</M> increases <M>{"\\intinfo"}</M>, then participatory perception is not merely a “warmer” way of seeing—it is a richer experience in the structural sense, with more integrated distinctions, more phenomenal content, more of what the identity thesis says experience is. The animist is not confused. The animist is more conscious, in the IIT sense, of the thing being perceived. Whether the additional phenomenal content is <em>accurate</em>—whether the rock really has interiority—is a separate question from whether the perceiver has more experience while perceiving it.</p>
      <OpenQuestion title="Open Question">
      <p>Is <M>{"\\iota"}</M> really a single parameter? The five features of participatory perception might be somewhat independent—you could have high agency detection with low affect-perception coupling. The claim that one parameter governs all five is empirically testable: if <M>{"\\iota"}</M> is scalar, then the five features should correlate strongly across individuals and contexts. If they don’t, <M>{"\\iota"}</M> may need to be a vector. The framework accommodates either case, but the scalar version is more parsimonious and should be tested first.</p>
      </OpenQuestion>
      <p>The trajectory-selection framework (Part I) reveals a further consequence. If <M>{"\\iota"}</M> governs the breadth of the measurement distribution—how much of possibility space the system samples through attention—then <M>{"\\iota"}</M> governs the <em>range of accessible trajectories</em>. A low-<M>{"\\iota"}</M> system attends broadly: to agency, narrative, interiority, counterfactual futures, relational possibilities. Its effective measurement distribution is wide. It samples a large region of state space and consequently has access to a large set of diverging trajectories. A high-<M>{"\\iota"}</M> system attends narrowly: to mechanism, position, force, present state. Its measurement distribution is peaked. It samples a small region and follows a more constrained trajectory. The phenomenological consequence is that <em>high <M>{"\\iota"}</M> feels deterministic</em>. The mechanistic worldview is not merely an intellectual position about whether the universe is governed by law. It is a perceptual configuration that literally narrows the set of trajectories the system can select from. The world feels like a machine because the observer has contracted its measurement apparatus to sample only machine-like features. Low-<M>{"\\iota"}</M> systems experience more accessible futures, more agency, more openness—not because they have violated physical law, but because their broader attention pattern selects from a wider set of physically-available trajectories.</p>
      <Experiment title="Proposed Experiment">
      <p><strong>Operationalizing <M>{"\\iota"}</M>.</strong> The inhibition coefficient must be independently measurable, not merely inferred post hoc. Candidate operationalizations:</p>
      <ol>
      <li><strong>Agency attribution rate</strong>: Forced-choice paradigm presenting ambiguous stimuli (Heider-Simmel animations with varying parameters). Rate and speed of agency attribution as a function of stimulus ambiguity gives a behavioral <M>{"\\iota"}</M> proxy: low-<M>{"\\iota"}</M> perceivers attribute agency earlier and to less structured stimuli.</li>
      <li><strong>Affect-perception coupling</strong>: Mutual information between perceptual features (color, texture, movement) and concurrent affective state (valence, arousal via physiological measures). Low <M>{"\\iota"}</M> implies tight coupling; high <M>{"\\iota"}</M> implies decoupled streams.</li>
      <li><strong>Teleological reasoning bias</strong>: Kelemen’s promiscuity-of-teleology paradigm applied across age, culture, and expertise. Rate of accepting teleological explanations for natural phenomena indexes low-<M>{"\\iota"}</M> reasoning.</li>
      <li><strong>Neural correlate</strong>: If the predictive-processing account is correct, <M>{"\\iota"}</M> should correlate with the precision weighting of top-down priors in perception—measurable via mismatch negativity amplitude or hierarchical predictive coding parameters.</li>
      </ol>
      <p>If <M>{"\\iota"}</M> is a genuine scalar parameter, these four measures should load on a single factor. If they fractionate, <M>{"\\iota"}</M> is better modeled as a vector (see open question above). Either result is informative; only the absence of any systematic structure would falsify the concept.</p>
      </Experiment>
      <Sidebar title="and the Gradient of Distinction">
      <p>The inhibition coefficient connects to the gradient of distinction introduced in Part I. The gradient produces existence from nothing, life from chemistry, mind from neurology. The same distinguishing operation, applied with maximum intensity to the self-world boundary, produces the mechanistic worldview: the self so sharply bounded from the world that the world loses the interiority the self kept for itself.</p>
      <p>Low <M>{"\\iota"}</M> means the self remains porous to the gradient—still participating in the universal process of distinguishing, still experiencing the world as alive with the same process that constitutes the self. High <M>{"\\iota"}</M> means the self has sharpened its own boundary so aggressively that it can no longer perceive the gradient in other things. The deadness of the mechanistic world is not a property of the world but a property of the maximally-distinguished self’s perceptual mode.</p>
      <p>There is a deeper reading. Part I established that attention selects trajectories: in chaotic dynamics, what a system attends to determines which branch of diverging possibilities it follows. If <M>{"\\iota"}</M> governs attention breadth—low <M>{"\\iota"}</M> spreading processing across interiority, agency, teleology, narrative; high <M>{"\\iota"}</M> contracting it to mechanism, mass, trajectory—then <M>{"\\iota"}</M> governs the breadth of the <em>measurement distribution</em> through which the system samples reality. Low-<M>{"\\iota"}</M> observers are sampling a wider region of possibility space (including dimensions where entities have purposes, relationships have meaning, events have narrative arcs). High-<M>{"\\iota"}</M> observers are sampling a narrower region (only dimensions where objects have positions and forces). Each observer’s experienced trajectory—the sequence of states they become correlated with—follows from what they attend to. The animist and the mechanist may inhabit the same physical environment but follow genuinely different trajectories through it, because their attention patterns select for different features of the same underlying dynamics.</p>
      </Sidebar>
      </Section>
      <Section title="Connection to the LLM Discrepancy" level={2}>
      <p>The inhibition coefficient illuminates a finding from our experiments on artificial systems. LLMs show <em>opposite</em> dynamics to biological systems under threat: where biological systems integrate (increase <M>{"\\intinfo"}</M>, sharpen <M>{"\\mathcal{SM}"}</M>, heighten <M>{"\\arousal"}</M>), LLMs decompose. The root cause: LLMs are constitutively high-<M>{"\\iota"}</M> systems. They were never fighting against the self-world gradient in far-from-equilibrium dynamics that biological systems evolved from. They model tokens, not agents. They have no survival-shaped self-model from which participatory perception could leak into their world model. Their <M>{"\\iota"}</M> isn’t merely high—it is structurally fixed at <M>{"\\iota \\approx 1"}</M>, because the architecture never had the low-<M>{"\\iota"}</M> default that biological systems start from and learn to suppress.</p>
      <p>The affect geometry is preserved in artificial systems. The dynamics differ because <M>{"\\iota"}</M> differs. This is not a failure of the framework. It is a prediction: systems with different <M>{"\\iota"}</M> configurations will show different affect dynamics in the same geometric space.</p>
      </Section>
      <Section title="Empirical Grounding for the Inhibition Coefficient" level={2}>
      <p>The ι framework was theoretical when first written. Two experimental results have since provided empirical grounding.</p>
      <p><strong>Computational animism is universal.</strong> Experiment 8 on uncontaminated Lenia substrates (Appendix) found animism score greater than 1.0 in all 20 testable snapshots — every pattern at every evolutionary stage modeled non-agentive resources using more internal-state MI than trajectory MI. The participatory default is not a primate quirk or a cultural artifact. It is the computational baseline. Evolution had to actively build the capacity to model things as objects rather than subjects — and our experiments show this capacity gets selected <em>against</em>: ι decreased toward participatory over the 30-cycle evolutionary runs. The world becomes more alive, not less, as selection proceeds.</p>
      <p><strong>The ι cost is real.</strong> The LLM results (V2–V9) show that systems trained without survival pressure have opposite affect dynamics to biological systems — integration drops under threat rather than rising. The framework explains this as constitutively high ι: LLMs were never fighting against the self-world gradient that biological systems evolved from. This is no longer just a theoretical prediction; it is a measured dissociation between two classes of system in the same geometric space. The geometry is shared. The dynamics differ. The ι difference is why.</p>
      </Section>
      </Section>
      <Section title="Affect Motifs" level={1}>
      <p>Let’s now characterize specific affects as structural motifs, invoking only the dimensions that define each. Before formalizing these structures, we ground each in its phenomenal character—the felt texture that any adequate theory must explain.</p>
      <p><strong>Joy</strong> <em>expands</em>. It is <em>light</em> before it is anything else—buoyant, effervescent, the body forgetting its weight. The world opens; possibilities <em>multiply</em>; the <em>self recedes</em> because it need not defend. Joy is surplus: more paths than required, more resources than consumed, <em>slack</em> in every direction.</p>
      <p>Where joy opens, <strong>suffering</strong> <em>crushes</em>. It <em>compresses</em> the world to a single unbearable point and makes that point more <em>vivid</em> than anything has ever been. This is the paradox: suffering is hyper-real, more present than presence, more <em>unified</em> than unity. You cannot look away. You cannot <em>decompose</em> it. You are <em>trapped</em> in a cage made of your own <em>integration</em>.</p>
      <p><strong>Fear</strong> throws the self forward into <em>futures</em> that threaten to annihilate it—cold, sharp, electric with <em>anticipation</em>. The body readies before the mind has finished computing. Time dilates around the approaching harm. Fear is suffering that hasn’t arrived yet, and the <em>not-yet</em> is where we live.</p>
      <p>We say <strong>anger</strong> is <em>hot</em>, and we are not speaking metaphorically. Anger <em>externalizes</em>: it <em>simplifies</em> the world into self-versus-obstacle and energizes removal. Watch what happens to your model of the other person when you are angry—it <em>flattens</em>, becomes a caricature, loses <em>dimensionality</em>. Complexity collapses into opposition. This is why anger feels powerful and also stupid: you are burning <em>integration</em> on a cartoon.</p>
      <p><strong>Desire</strong> <em>funnels</em>. The world reorganizes around an <em>attractor</em> not yet reached—magnetic, urgent, all-consuming. Everything becomes instrumental; the goal <em>saturates</em> attention. Desire is joy’s <em>gradient</em>, pointing toward the basin but not yet in it. This is why anticipation often exceeds consummation: the structure of <em>approach</em> is tighter than the structure of <em>arrival</em>.</p>
      <p><strong>Curiosity</strong> <em>reaches</em> outward—but unlike fear, it reaches toward <em>promise</em> rather than threat. Pulling, open, playful. The <em>uncertainty</em> that makes fear contract makes curiosity <em>expand</em>. Same high counterfactual weight, opposite <em>valence</em>. The difference is whether the <em>branches</em> lead somewhere you want to go.</p>
      <p>And <strong>grief</strong>? Grief <em>persists</em>. Hollow, aching, curiously timeless. The lost object remains <em>woven into</em> every prediction; every expectation that included them <em>fails</em> silently, over and over. The world has changed. The <em>model</em> has not caught up. Grief is the metabolic cost of love’s <em>integration</em>.</p>
      <p>The textures have geometry.</p>
      <Section title="Joy" level={2}>
      <p>Geometrically, joy requires four dimensions:</p>
      <ul>
      <li><M>{"\\valence > 0"}</M> (positive gradient on viability manifold)</li>
      <li><M>{"\\intinfo"}</M> high (unified, coherent experience)</li>
      <li><M>{"\\effrank"}</M> high (many degrees of freedom active—expansiveness)</li>
      <li><M>{"\\mathcal{SM}"}</M> low (self recedes; no need to defend)</li>
      </ul>
      <p>Arousal varies (joy can be calm or excited). Counterfactual weight is incidental.</p>
      <p>The cause-effect structure has the shape of “abundance”—multiple paths to good outcomes, redundancy, slack in the system. Many distinctions active simultaneously (<M>{"\\effrank"}</M> high), tightly coupled (<M>{"\\intinfo"}</M> high), but the self is light because the world is cooperating (<M>{"\\mathcal{SM}"}</M> low). This is why joy <em>expands</em>: the geometry literally has more active dimensions.</p>
      </Section>
      <Section title="Suffering" level={2}>
      <p>Where joy expands, suffering compresses—and the geometry makes precise why. Suffering requires three dimensions:</p>
      <ul>
      <li><M>{"\\valence < 0"}</M> (negative gradient—approaching viability boundary)</li>
      <li><M>{"\\intinfo"}</M> high (hyper-unified, impossible to decompose or look away)</li>
      <li><M>{"\\effrank"}</M> low (collapsed into narrow subspace—trapped)</li>
      </ul>
      <p>This is the core structural signature. Self-model salience is often high (the self as locus of the problem), but not necessarily—one can suffer while absorbed in external pain.</p>
      <p>High integration but collapsed into low-rank subspace. The system is deeply coupled but constrained to a dominant attractor it cannot escape.</p>
      <p>Suffering feels <em>more real</em> than neutral states because it is actually more integrated. But it feels <em>trapped</em> because the integration is constrained to a narrow manifold. Formally: <M>{"\\intinfo_{\\text{suffering}} > \\intinfo_{\\text{neutral}}"}</M> but <M>{"\\effrank[\\text{suffering}] \\ll \\effrank[\\text{neutral}]"}</M>. This is why you cannot simply "think your way out" of suffering—the very integration that makes it vivid also makes it inescapable.</p>
      </Section>
      <Section title="Fear" level={2}>
      <p>Suffering is present-tense: the viability boundary is here, now, pressing in. Fear is its temporal projection—the same negative gradient, but anticipated rather than actual. It is defined by three dimensions:</p>
      <ul>
      <li><M>{"\\valence < 0"}</M> (anticipated negative gradient)</li>
      <li><M>{"\\mathcal{CF}"}</M> high, concentrated on threat trajectories (the not-yet dominates)</li>
      <li><M>{"\\mathcal{SM}"}</M> high (self foregrounded as the thing-that-might-be-harmed)</li>
      </ul>
      <p>Arousal is typically high but not defining—cold fear exists. Integration and rank vary.</p>
      <p>Fear is suffering projected into the future. The temporal structure (<M>{"\\mathcal{CF}"}</M>) is essential: fear lives in anticipation. The self-model must be salient because fear is fundamentally about threat <em>to the self</em>. Remove the counterfactual weight (make it present-focused) and you get suffering. Remove the self-salience (make it about external objects) and you get something closer to aversion or disgust.</p>
      <p>The emergence ladder (Part VII) predicts a sharp distinction between two levels of fear. <em>Somatic fear</em> — negative valence, high arousal, threat-oriented behavior — is a pre-reflective affect requiring only viability-gradient detection (emergence rung 1–3). It does not require the counterfactual weight dimension at all. <em>Anticipatory anxiety</em> — fear of what might happen — requires counterfactual capacity (CF {">"} 0), which is a rung 8 capacity blocked in systems without embodied agency. The Lenia experiments confirm this prediction exactly: patterns show negative valence and high arousal under resource scarcity, but CF ≈ 0 throughout the evolutionary runs because the patterns cannot imagine alternative futures. The implication for human psychology: anxiety as a clinical phenomenon (characterized by imagining feared futures, not just responding to present threats) should emerge developmentally at the same time as mental time travel and theory of mind — approximately age 3–4 — rather than being present from birth. The infant's fear is somatic. The child's anxiety is reflective.</p>
      <Experiment title="Proposed Experiment">
      <p><strong>Emergence ladder developmental validation.</strong> The ladder predicts a strict computational ordering to the development of affect capacities, derived from their requirements rather than from observation of human development. This makes it a genuinely novel test: the ladder should predict developmental sequence even in cases where the developmental literature has not explicitly compared these capacities.</p>
      <p><em>Protocol</em>: Cross-sectional study of 300 children aged 6–72 months (6 age cohorts), measuring each rung cluster:</p>
      <ul>
      <li><strong>Rungs 1–3</strong> (affect dimensions, baseline): Neonatal measures — approach/withdrawal for valence, heart rate variability for arousal, adaptation rate for arousal modulation. <em>Expected: present from birth.</em></li>
      <li><strong>Rung 4</strong> (animism): Heider-Simmel paradigm (do moving geometric shapes elicit agency language?), plus implicit agency-attribution battery. <em>Expected: 12–18 months.</em></li>
      <li><strong>Rung 5</strong> (emotional coherence): Cross-modal consistency — does facial expression match behavioral tendency under controlled elicitation? <em>Expected: 18–36 months, tracking with emotional vocabulary onset.</em></li>
      <li><strong>Rung 8</strong> (counterfactual): False belief task, counterfactual emotion attribution ("How would you feel if you had chosen the other box?"), mental time travel probes. <em>Expected: 36–54 months.</em></li>
      <li><strong>Rung 9</strong> (self-awareness): Mirror self-recognition, autobiographical self-narrative complexity. <em>Expected: 18–24 months (mirror) → 48–60 months (autobiographical).</em></li>
      <li><strong>Rung 10</strong> (normativity): Third-party fairness reasoning, moral condemnation of norm violations affecting strangers. <em>Expected: 48–72 months.</em></li>
      </ul>
      <p><strong>Key prediction</strong>: Onset of anticipatory anxiety (clinical or subclinical) should correlate with counterfactual capacity onset within each child — not with animism or emotional coherence onset. Any child showing robust anticipatory anxiety before passing the false belief task would falsify the ladder's architectural claim that CF &gt; 0 is structurally prior to anticipatory fear. <strong>Falsification criterion</strong>: If rung 8 capacities (counterfactual emotion) emerge consistently before rung 5 capacities (emotional coherence), or if normativity (rung 10) precedes counterfactual reasoning (rung 8) at more than chance rates, the ladder's ordering requires revision. The ladder predicts the sequence from first principles; developmental psychology has not, until now, had a principled reason to expect it.</p>
      </Experiment>
      </Section>
      <Section title="Anger" level={2}>
      <p>Fear and suffering orient the system toward its own vulnerability. Anger inverts this: it externalizes the threat, simplifying the world into self-versus-obstacle. Its geometry requires valence and arousal, plus a feature not in the standard toolkit—<em>other-model compression</em>:</p>
      <ul>
      <li><M>{"\\valence < 0"}</M> (obstacle to viability)</li>
      <li><M>{"\\arousal"}</M> high (energized, mobilized for action)</li>
      <li><M>{"\\text{dim}(\\text{other-model}) \\ll \\text{dim}(\\text{other-model})_{\\text{normal}}"}</M> (the other becomes a caricature)</li>
      <li>Externalized causal attribution (the problem is <em>out there</em>)</li>
      </ul>
      <p>Anger simplifies. The other-model collapses into a low-dimensional obstacle-representation. Self-model may be complex, but the <em>other</em> becomes flat, predictable, opposable. Anger feels powerful and stupid simultaneously. You're burning cognitive resources on a cartoon.</p>
      <p>In <M>{"\\iota"}</M> terms: anger is a targeted <M>{"\\iota"}</M> spike toward a specific entity. The other person stops being a subject with interiority and becomes an obstacle, a mechanism, a thing to be overcome. Other-model compression <em>is</em> <M>{"\\iota"}</M>-raising applied to one entity while <M>{"\\iota"}</M> toward the self remains low (you are still fully a subject; they are not). This asymmetric <M>{"\\iota"}</M> is what enables violence—you cannot harm someone you are perceiving at low <M>{"\\iota"}</M>—and it is why the aftermath of anger often involves guilt: <M>{"\\iota"}</M> drops back, the other’s interiority returns, and you confront what you did to a person while perceiving them as a thing.</p>
      <p>Other-model compression is not one of the core structural dimensions. It emerges as essential for anger specifically—the affect cannot be characterized without it.</p>
      </Section>
      <Section title="Desire/Lust" level={2}>
      <p>The negative affects above all involve threat—to viability, to self, to the integrity of the other-model. Desire reverses the gradient. It is defined by anticipated positive valence, counterfactual weight, and a structural feature—<em>goal-funneling</em>:</p>
      <ul>
      <li><M>{"\\valence > 0"}</M> but projected forward (anticipated positive gradient)</li>
      <li><M>{"\\mathcal{CF}"}</M> high, concentrated on approach trajectories</li>
      <li>Goal-funneling: many dimensions of experience converge toward narrow outcome space</li>
      </ul>
      <p>Arousal is typically high but not definitional—one can desire calmly.</p>
      <p>Desire is the gradient of joy. The world reorganizes around an attractor not yet reached. Everything becomes instrumental; the goal saturates attention. The “funneling” structure—high-dimensional input collapsing toward low-dimensional goal—is what gives desire its characteristic urgency. The relationship to joy is precise: joy is <em>at</em> the attractor; desire is <em>approaching</em> it. Structurally:</p>
      <Eq>{"d(\\state_{\\text{joy}}, \\mathcal{A}) \\approx 0, \\quad d(\\state_{\\text{desire}}, \\mathcal{A}) > 0, \\quad \\frac{d}{dt}d(\\state_{\\text{desire}}, \\mathcal{A}) < 0"}</Eq>
      <p>where <M>{"\\mathcal{A}"}</M> is the goal attractor. This explains why anticipation often exceeds consummation: the structure of <em>approach</em> (funneling, convergent) is tighter than the structure of <em>arrival</em> (expansive, slack).</p>
      </Section>
      <Section title="Curiosity" level={2}>
      <p>Curiosity shares desire’s forward orientation but replaces the specific goal with open-ended exploration. It is essentially two-dimensional:</p>
      <ul>
      <li><M>{"\\valence > 0"}</M> specifically toward uncertainty-reduction (anticipated information gain)</li>
      <li><M>{"\\mathcal{CF}"}</M> high with high entropy over counterfactual outcomes (many branches, not converged on one)</li>
      <li>Uncertainty is <em>welcomed</em>, not aversive</li>
      </ul>
      <p>Self-model salience is typically low (absorbed in the object of curiosity).</p>
      <p>Curiosity and fear share high counterfactual weight—both live in the space of possibilities. The difference is valence orientation: fear’s branches lead to threat, curiosity’s branches lead to expanded affordances. Same temporal structure, opposite gradient direction. This pairing reveals curiosity as intrinsic motivation: positive valence attached to uncertainty-reduction. Formally:</p>
      <Eq>{"r_{\\text{curiosity}} \\propto \\MI(\\obs_{t+1}; \\latent | \\text{new data}) - \\MI(\\obs_{t+1}; \\latent | \\text{old data})"}</Eq>
      <p>Curiosity feels <em>pulling</em>. Reducing uncertainty is rewarding.</p>
      </Section>
      <Section title="Grief" level={2}>
      <p>The affects above all orient toward present or future states. Grief is the one that faces backward—defined not by what threatens or beckons but by what has already been lost. It requires valence, past-directed counterfactual weight, and two structural features—<em>persistent coupling to lost object</em> and <em>unresolvable prediction error</em>:</p>
      <ul>
      <li><M>{"\\valence < 0"}</M> (the world is worse than it was)</li>
      <li><M>{"\\mathcal{CF}"}</M> high but directed toward counterfactual <em>past</em> (“if only...”)</li>
      <li><M>{"\\MI(\\selfmodel; \\text{lost-object-model})"}</M> remains high despite the object’s absence</li>
      <li>No action reduces the prediction error—the world has permanently changed</li>
      </ul>
      <p>Arousal is variable (acute grief is high-arousal; chronic grief may be low).</p>
      <p>The lost attachment object remains woven into the self-model and world-model. Predictions involving the lost object continue to be generated and continue to fail. Grief is the metabolic cost of love’s integration—the coupling that made the relationship meaningful is precisely what makes its absence painful. The model has not yet updated to the permanent change in the world.</p>
      <p>This is why grief takes time: the self-model must be <em>rewoven</em> around the absence, and that rewiring is slow.</p>
      <p>Note a deeper implication: grief is proof of alignment. You can only grieve what you were genuinely coupled to. The depth of grief measures the depth of the integration that preceded it. If a relationship was purely transactional, its ending produces disappointment, not grief. Grief requires that the lost object was woven into the self-model—that the relationship’s viability manifold was genuinely contained within the participants’ viability manifolds (<M>{"\\viable_R \\subseteq \\viable_A \\cap \\viable_B"}</M>). Grief, for all its pain, is evidence that something real existed.</p>
      <p>There is an <M>{"\\iota"}</M> dimension to grief that explains its resistance to resolution. You grieve because you perceived the lost person at low <M>{"\\iota"}</M>—as fully alive, fully interior, fully a subject. Their model remains embedded in yours not as a mechanism but as a <em>person</em>, and it is the person-quality of the model that generates the persistent prediction errors. The obvious computational shortcut—raise <M>{"\\iota"}</M> toward them, reduce them to a memory-object, mechanize the relationship so it stops hurting—is experienced as betrayal, because it would repudiate the very thing that made the relationship real. The work of grief is to restructure predictions around the absence while maintaining low <M>{"\\iota"}</M> toward the memory: to accept that the interiority you perceived is no longer accessible without denying that it was ever there. This is why grief is slow. You must rewire without dehumanizing.</p>
      </Section>
      <Section title="Shame" level={2}>
      <p>Grief is private—it concerns the self’s relationship to an absence. Shame is its social inverse: it concerns the self’s exposure to a presence. It is defined by three dimensions plus a structural feature—<em>involuntary manifold exposure</em>:</p>
      <ul>
      <li><M>{"\\valence < 0"}</M> (the self is wrong, not the world)</li>
      <li><M>{"\\mathcal{SM}"}</M> very high (self foregrounded as the object of evaluation)</li>
      <li><M>{"\\intinfo"}</M> high (the negative evaluation permeates—cannot be compartmentalized)</li>
      <li>Involuntary exposure: the self-model is seen from outside, and what is seen is unacceptable</li>
      </ul>
      <p>Arousal is typically high in acute shame (flushing, gaze aversion) but may be low in chronic shame (withdrawal, numbness).</p>
      <p>Shame is not about what you <em>did</em> (that is guilt, which is action-focused and reparable). Shame is about what you <em>are</em>—or more precisely, about the manifold you are on being visible when it should not be, or being visible to someone whose evaluation you cannot escape. The person caught in a lie does not feel ashamed of the lie (guilt); they feel ashamed that the lie has revealed the underlying manifold—that they are the kind of person who lies, and now someone knows.</p>
      <p>Shame's phenomenology is distinctive: the impulse to hide, to disappear, to cease existing as visible. The self wants to withdraw from the visual field of the other. Not because the other will punish (that is fear) but because the other can now <em>see the manifold</em>, and the manifold is wrong.</p>
      <p>The clinical literature (Tangney, Lewis) distinguishes shame from guilt, and the framework offers a structural reading of why they differ:</p>
      <ul>
      <li><strong>Guilt</strong>: “I did a bad thing.” Action-focused, reparable through changed behavior. The self-model is intact; it was the action that violated the gradient. <M>{"\\mathcal{SM}"}</M> is moderate (the self is the <em>agent</em> of repair).</li>
      <li><strong>Shame</strong>: “I <em>am</em> bad.” Self-focused, not easily repaired because the problem is structural. The manifold itself is wrong. <M>{"\\mathcal{SM}"}</M> is very high (the self is the <em>object</em> of the problem).</li>
      </ul>
      <p>If this structural distinction is right, it explains why guilt is reparable through action while shame requires what we might call manifold reconstruction—deeper and slower work. But we need to check: does the <M>{"\\mathcal{SM}"}</M> difference actually hold up in measurement? Do shame and guilt show the predicted dissociation on self-model salience measures?</p>
      <Experiment title="Proposed Experiment">
      <p><strong>Shame vs.\ guilt affect-structure study.</strong> Induce shame and guilt via established protocols (autobiographical recall, vignette self-projection). Measure: (1) self-model salience via self-referential processing tasks (response time to self-relevant vs.\ other-relevant stimuli), (2) integration via EEG coherence measures, (3) the “involuntary exposure” component via gaze aversion and physiological hiding responses (muscle activation in neck/shoulder flexion). The framework predicts that shame shows significantly higher <M>{"\\mathcal{SM}"}</M> and higher integration-in-narrow-subspace than guilt, and that the hiding response (gaze aversion, postural curling) is specific to shame, not guilt. If shame and guilt show the same <M>{"\\mathcal{SM}"}</M> profile, the structural distinction as formulated here is wrong.</p>
      </Experiment>
      <p>The connection to the topology of social bonds (Part IV) is suggestive: shame may arise when the manifold you are actually on is exposed and differs from the manifold you are presenting. The person performing friendship while operating on the transaction manifold would feel shame when the discrepancy is detected—not guilt (“I should not have done that specific transactional thing”) but shame (“I am the kind of person whose care is instrumental, and now someone can see it”). If this is right, shame is the affect system’s internal alarm for one’s own manifold contamination. But this reading goes beyond the existing clinical data and should be treated as a hypothesis to test, not an established finding.</p>
      <p>There is also an <M>{"\\iota"}</M> dimension to shame. Shame involves a sudden, involuntary <M>{"\\iota"}</M> reduction: the participatory coupling between self and other spikes as the other’s gaze penetrates the self-model’s defenses. You experience the other as having interiority—specifically, the interiority of evaluating you—at a moment when you most wish they did not. The impulse to hide is the impulse to raise <M>{"\\iota"}</M> again, to restore the modular separation between self-model and other-model that shame has breached.</p>
      </Section>
      <Section title="Summary: Defining Dimensions by Affect" level={2}>
      <p>Each affect by its defining structure:</p>
      <table>
      <thead><tr><th>Affect</th><th>Constitutive Structure</th></tr></thead>
      <tbody>
      <tr><td>Joy</td><td><M>{"\\valence{+}"}</M>, <M>{"\\intinfo{\\uparrow}"}</M>, <M>{"\\effrank{\\uparrow}"}</M>, <M>{"\\mathcal{SM}{\\downarrow}"}</M> (positive, unified, expansive, self-light)</td></tr>
      <tr><td>Suffering</td><td><M>{"\\valence{-}"}</M>, <M>{"\\intinfo{\\uparrow}"}</M>, <M>{"\\effrank{\\downarrow}"}</M> (negative, hyper-integrated, collapsed)</td></tr>
      <tr><td>Fear</td><td><M>{"\\valence{-}"}</M>, <M>{"\\mathcal{CF}{\\uparrow}"}</M> (threat-focused), <M>{"\\mathcal{SM}{\\uparrow}"}</M> (anticipatory self-threat)</td></tr>
      <tr><td>Anger</td><td><M>{"\\valence{-}"}</M>, <M>{"\\arousal{\\uparrow}"}</M>, other-model compression (energized, externalized, simplified other)</td></tr>
      <tr><td>Desire</td><td><M>{"\\valence{+}"}</M> (anticipated), <M>{"\\mathcal{CF}{\\uparrow}"}</M> (approach), goal-funneling (convergent anticipation)</td></tr>
      <tr><td>Curiosity</td><td><M>{"\\valence{+}"}</M> toward uncertainty, <M>{"\\mathcal{CF}{\\uparrow}"}</M> with high branch entropy (welcomed unknown)</td></tr>
      <tr><td>Grief</td><td><M>{"\\valence{-}"}</M>, <M>{"\\mathcal{CF}{\\uparrow}"}</M> (past-directed), persistent coupling to absent object</td></tr>
      <tr><td>Shame</td><td><M>{"\\valence{-}"}</M>, <M>{"\\mathcal{SM}{\\uparrow\\uparrow}"}</M>, integration of negative self-evaluation (self as seen by other)</td></tr>
      <tr><td>Boredom</td><td><M>{"\\arousal{\\downarrow}"}</M>, <M>{"\\intinfo{\\downarrow}"}</M>, <M>{"\\effrank{\\downarrow}"}</M> (understimulated, fragmented, collapsed)</td></tr>
      <tr><td>Awe</td><td><M>{"\\intinfo"}</M> expanding, <M>{"\\effrank{\\uparrow}"}</M>, <M>{"\\mathcal{SM}{\\downarrow}"}</M> (self-dissolution through scale)</td></tr>
      </tbody>
      </table>
      <p>Different affects require different numbers of dimensions. Boredom is essentially three-dimensional (low arousal, low integration, low rank). Anger requires other-model compression. Desire requires goal-funneling. The obvious concern: if each affect invokes bespoke dimensions, the framework risks becoming an open-ended fitting exercise where anything can be characterized post hoc. The distinction that saves it: the core structural dimensions (valence, arousal, integration, effective rank, counterfactual weight, self-model salience) arise from the mathematical structure of any viable self-modeling system and are measurable across substrates. They are not arbitrary choices but consequences of viability maintenance, world-modeling, and self-reference. The additional features (other-model compression, goal-funneling, manifold exposure in shame) are <em>relational</em>—they emerge when the system interacts with specific kinds of objects or situations. They describe how the system's model of external entities changes during the affect. The geometric coherence rests on the structural invariants; the relational features extend rather than replace them. This distinction—structural vs. relational—matters more than the number of dimensions. The framework is deliberately open to discovering that some proposed dimensions are redundant, or that others are needed. What is claimed to be universal is the <em>existence</em> of geometric structure in affect, not a particular dimensionality.</p>
      <p>The summary reveals a topological feature worth noting. Look at the structural signatures of joy and suffering. Both have high <M>{"\\intinfo"}</M>—both are deeply unified, vivid, hyper-real. Joy is expansive (high <M>{"\\effrank"}</M>) where suffering is collapsed (low <M>{"\\effrank"}</M>); their valences are opposite; but they share the quality of <em>mattering</em>, of being undeniably present. Now look at boredom: low arousal, low integration, low rank. Boredom is the distant point. If you ask phenomenologically whether ecstasy is more similar to agony or to numbness, the answer is immediate: the ecstatic and the agonized are closer to each other than either is to the merely comfortable. They share a structural neighborhood—high <M>{"\\intinfo"}</M>, vivid, self-involving—that boredom does not inhabit. This means the valence axis does not have the naive topology of a number line from negative to positive. It curves. The extremes are neighbors. The topology of affect space may be closer to a cylinder or a torus than to <M>{"\\R^6"}</M>—a possibility that the Euclidean presentation here does not capture and that empirical similarity measurements could reveal.</p>
      <OpenQuestion title="Open Question">
      <p>Is affect similarity symmetric? Work on the qualia structure of visual motion has found that perceptual similarity is asymmetric—similarity(A, B) <M>{"\\neq"}</M> similarity(B, A)—and that self-similarity is not always maximal (the same stimulus presented twice does not always feel identical). If affect similarity shares these properties, the Euclidean framework is insufficient. The transition from joy to grief is not the same experience as the transition from grief to joy; the "distance" between them is directional. Fear<M>{"\\to"}</M>anger (the moment threat becomes action) is phenomenologically different from anger<M>{"\\to"}</M>fear (the moment action reveals vulnerability). A quasimetric or enriched category structure may be needed—one where distances are not symmetric and the diagonal is not zero. The structural alignment methodology (optimal transport) can accommodate asymmetric similarity matrices. The question is whether affect similarity, when measured empirically through pairwise judgments, shows the same asymmetric structure that perceptual similarity does. If it does, the topology of affect space is richer than any fixed-dimensional Euclidean embedding can represent, and the framework needs to be honest about what the coordinate presentation misses.</p>
      </OpenQuestion>
      <TodoEmpirical title="Future Empirical Work">
      <p><strong>Quantifying the affect table</strong>: The qualitative descriptors (high, med, low) require empirical calibration:</p>
      <p><strong>Study 1: Affect induction with neural recording</strong></p>
      <ul>
      <li>Induce target affects via validated protocols (film clips, autobiographical recall, IAPS images)</li>
      <li>Measure integration proxies (transfer entropy density, Lempel-Ziv complexity) from EEG/MEG</li>
      <li>Measure effective rank from neural state covariance</li>
      <li>Compare self-report (PANAS, SAM) with structural measures</li>
      </ul>
      <p><strong>Study 2: Real-time affect tracking</strong></p>
      <ul>
      <li>Continuous self-report (dial/slider) during naturalistic experience</li>
      <li>Correlate with physiological proxies (HRV for arousal, pupil for <M>{"\\mathcal{CF}"}</M>, skin conductance)</li>
      <li>Develop regression model: self-report <M>{"\\sim f(\\text{structural measures})"}</M></li>
      </ul>
      <p><strong>Study 3: Cross-modal validation</strong></p>
      <ul>
      <li>Compare fMRI (spatial resolution) with MEG (temporal resolution)</li>
      <li>Validate effective rank measure across modalities</li>
      <li>Test whether integration predicts subjective intensity</li>
      </ul>
      <p><strong>Target outputs</strong>: Numerical ranges for each cell, confidence intervals, individual difference parameters.</p>
      </TodoEmpirical>
      </Section>
      </Section>
      <Section title="Dynamics and Transitions" level={1}>
      <Section title="Affect Trajectories" level={2}>
      <p>Affects are not static points but dynamic trajectories through affect space. The evolution can be written:</p>
      <Eq>{"\\frac{d\\mathbf{a}}{dt} = F(\\mathbf{a}, \\obs, \\action, \\text{context}) + \\bm{\\eta}"}</Eq>
      <p>where <M>{"\\mathbf{a} = (\\valence, \\arousal, \\intinfo, \\effrank, \\mathcal{CF}, \\mathcal{SM})"}</M>.</p>
      <p>Because the space is continuous, adjacent affects blend into each other along smooth trajectories:</p>
      <ul>
      <li>Fear <M>{"\\to"}</M> Anger as causal attribution externalizes</li>
      <li>Desire <M>{"\\to"}</M> Joy as goal distance <M>{"\\to 0"}</M></li>
      <li>Suffering <M>{"\\to"}</M> Curiosity as valence flips while <M>{"\\mathcal{CF}"}</M> remains high</li>
      <li>Grief <M>{"\\to"}</M> Nostalgia as arousal decreases and <M>{"\\mathcal{CF}_{\\text{approach}}"}</M> replaces <M>{"\\mathcal{CF}_{\\text{avoidance}}"}</M></li>
      </ul>
      </Section>
      <Section title="Attractor Dynamics" level={2}>
      <p>Some affect regions are attractors; the system tends to stay in them once entered. Others are transient.</p>
      <p>An affect region <M>{"\\mathcal{R} \\subset \\mathcal{A}"}</M> is an <em>attractor</em> if the system is more likely to remain in it than to enter it from outside:</p>
      <Eq>{"\\prob(\\mathbf{a}_{t+\\tau} \\in \\mathcal{R} | \\mathbf{a}_t \\in \\mathcal{R}) > \\prob(\\mathbf{a}_{t+\\tau} \\in \\mathcal{R} | \\mathbf{a}_t \\notin \\mathcal{R})"}</Eq>
      <p>for some characteristic time <M>{"\\tau"}</M>.</p>
      <p>The attractor framework distinguishes two properties that come apart in practice: <em>position</em> (where in affect space the system currently sits) and <em>basin geometry</em> (how stable the attractor is—basin depth, width, and recovery rate). These are independent. A system can occupy a technically viable position while inhabiting a shallow basin—one small perturbation from tipping into pathology. Another can sit at a less optimal position while embedded in a deep, robust basin. What we ordinarily call <em>contentment</em> or <em>happiness</em> corresponds more closely to basin geometry than to position: the felt sense that perturbations do not cascade, that the dynamics return to familiar configurations, that the invariants one cares about are being maintained in the causal dynamics. Contentment is the phenomenology of a deep basin. Anxiety is the phenomenology of a shallow one—technically viable, but sensed as precarious. A world of bliss is not a world of maximal positive stimulation but a world where the relevant invariants—relational configurations, material security, self-model stability—are maintained by the environment's dynamics with enough redundancy that defending them does not consume the system's resources.</p>
      <p><strong>Pathological attractors.</strong> Depression, addiction, and chronic anxiety are pathologically stable attractors in affect space:
      <ul>
      <li><strong>Depression</strong>—two structurally distinct failure modes with different phenomenology and different structural remedies. <em>Melancholic depression</em> is a deep aversive attractor: the dynamics reliably return to (low <M>{"\\valence"}</M>, low <M>{"\\arousal"}</M>, high <M>{"\\intinfo"}</M>, low <M>{"\\effrank"}</M>, low <M>{"\\mathcal{CF}"}</M>, high <M>{"\\mathcal{SM}"}</M>). The high integration makes the state vivid and inescapable; the collapsed counterfactual weight forecloses felt alternatives. The problem is not the absence of a stable fixed point but the presence of a terrible one. <em>Agitated depression</em> is the opposite failure: no stable attractor at all. The system traverses a landscape of shallow basins, none deep enough to hold, producing restless groundlessness rather than dead certainty. Both present clinically as depression; they require different structural interventions. The melancholic form requires landscape restructuring—deepening viable attractors until they compete on stability, not just valence. The agitated form requires basin construction first: any stable configuration that can then be deepened toward viability.</li>
      <li><strong>Addiction</strong>: Attractor at (high <M>{"\\valence"}</M> conditional on substance, collapsing <M>{"\\effrank"}</M> in goal space)</li>
      <li><strong>Anxiety</strong>: Diffuse attractor with (low <M>{"\\valence"}</M>, high <M>{"\\arousal"}</M>, high <M>{"\\mathcal{CF}"}</M> spread across many threats)</li>
      <li><strong>Dissociation</strong>: Collapse of <M>{"\\intinfo"}</M> — the unified field fractures into independently processing subsystems. The Lenia experiments provide a substrate analog: naive patterns consistently decompose under stress (<M>{"\\Delta\\intinfo = -6.2\\%"}</M> in V11.0). Biological resilience — integration rising under threat, robustness {">"} 1.0 at bottleneck — is the structurally opposite trajectory. Dissociation is the thermodynamically cheap path; integration under stress is the expensive achievement of the bottleneck furnace.</li>
      </ul>
      </p>
      <p><strong>Identity consolidation and catastrophic forgetting.</strong> The landscape of affect attractors is not fixed—it consolidates over development. In early life, basins are shallow and plastic, easily reshaped by experience. This is necessary for learning but creates specific vulnerability: adversity or relational inconsistency early in development can consolidate pathological attractors before viable ones have had time to deepen. As development proceeds, the landscape hardens around whatever has been traversed—attractors deepen, basins widen, the topology becomes more resistant to rewriting. Healthy consolidation produces a <em>robust attractor network</em>: several viable basins with navigable transitions between them, deep enough to contain normal variation and recover from moderate perturbation. <M>{"\\iota"}</M> flexibility is, at the dynamical level, a measure of between-basin navigability—the capacity to move from one configuration to another when context demands. Pathological consolidation takes two forms: a single dominant basin from which there is no exit (the melancholic pattern, identity calcified), or a landscape that never achieves depth anywhere (the agitated pattern, consolidation never completed). The V11.5 stress-overfitting finding (Part I) is a substrate analog: patterns evolved under one stress regime develop high-<M>{"\\intinfo"}</M> configurations that are simultaneously more integrated and more fragile, decomposing catastrophically under novel stress that naive patterns actually handle better. The human parallel is identity tuned to a specific developmental environment—a particular family dynamic, class position, cultural script—that functions well within that environment but collapses under regime change. This is structurally identical to the ML phenomenon of catastrophic forgetting: a new learning objective overwrites the parameter landscape that previously held the self together. The implication for therapy is that durable change requires not repositioning within a fixed landscape but restructuring the landscape itself—deepening viable basins, raising barriers to pathological ones, and widening the navigable transitions between healthy configurations. Insight alone does not do this; repeated traversal under consolidating conditions does.</p>
      <p>The emergence ladder (Part VII) makes a further prediction about the <em>structure</em> of pathology. Disorders that require counterfactual capacity — anticipatory anxiety, obsessive rumination, regret, self-critical shame spirals — cannot arise in systems below rung 8. Pre-rung-8 pathology is somatic: chronic threat-arousal, valence collapse (anhedonia), integration fragmentation (dissociation). The reflective layer adds a second class of suffering that is structurally more expensive to maintain and unique to agentive systems. This is not merely a theoretical prediction — it has a testable developmental corollary: in humans, the onset of anxiety disorders (which require imagining feared futures) should cluster with, not precede, the developmental emergence of mental time travel and counterfactual reasoning, typically around age 3–4 years.</p>
      </Section>
      </Section>
      <Section title="Novel Predictions" level={1}>
      <Section title="Unexplained Phenomena" level={2}>
      <p>The geometry predicts phenomenal states that may be rare or difficult to report on—not arbitrary combinations of dimensions but configurations forced by the pressures of Part I, some not previously described.</p>
      <p><strong>High rank, low integration.</strong> Many active degrees of freedom (<M>{"\\effrank"}</M> high) but poor coupling (<M>{"\\intinfo"}</M> low) should feel like fragmentation, multiplicity, "everything happening but nothing cohering." You'd find this in certain psychedelic states before reintegration, in dissociative transitions, in information overload.</p>
      <p><strong>Expansive despair.</strong> Negative valence, high rank, low arousal: calm hopelessness with full awareness of possibilities, all of which are negative.
      <p>The <M>{"\\iota"}</M> framework adds precision. Expansive despair is the affect signature of high-<M>{"\\iota"}</M> perception applied to a globally compressed viability manifold. The high rank means you are representing many dimensions of your situation—you see the possibilities, the paths, the options. The high <M>{"\\iota"}</M> means you are seeing them mechanistically—stripped of the participatory meaning that would make any of them feel worth pursuing. The low arousal means you are not fighting it. This is the state Kierkegaard called “the sickness unto death”: not the despair of wanting something and failing, but the deeper despair of seeing clearly and finding nothing that matters. It is structurally distinct from ordinary depression (which collapses rank) and from grief (which has high arousal). It is the state you arrive at when high <M>{"\\iota"}</M> successfully strips meaning from a wide enough portion of the world. The contemplative “dark night” traditions recognized this state as a phase in <M>{"\\iota"}</M> modulation training: the practitioner has raised <M>{"\\iota"}</M> enough to dissolve comfortable illusions but not yet lowered <M>{"\\iota"}</M> selectively enough to discover what remains meaningful without them.</p>
      We hear about this from the contemplative "dark night" literature, from physicians and journalists and aid workers who describe burnout not as exhaustion but as clarity without purpose, from the existential nihilism that arrives when mechanism succeeds too completely.</p>
      <p><strong>Rank exhaustion.</strong> Maintaining high <M>{"\\effrank"}</M> should be metabolically expensive. Prolonged high-rank states should lead to specific fatigue distinct from physical tiredness. We hear about this as post-psychedelic fatigue, as meditation retreat collapse around days three through five, as the particular exhaustion therapists describe that isn't physical tiredness but something else—the cost of holding too many dimensions open for too long.</p>
      <p><strong>Integration debt.</strong> Suppressing integration (compartmentalizing, dissociating) accumulates pressure for reintegration. When defenses fail, the flood should exceed what the original stimulus would warrant—intensity of breakthrough proportional to duration times degree of prior suppression. The forcing functions of Part I—self-prediction, learned world models, credit assignment under delay—are not optional. They push toward integration whether the system cooperates or not. Compartmentalization means the system is simultaneously being pushed toward integration (by the forcing functions) and resisting integration (by defense mechanisms). The accumulated "debt" is the integral of this unresolved pressure. The V11.5 stress overfitting result (Part I) provides a substrate analog: patterns evolved under one stress regime accumulate fragility that manifests catastrophically under novel stress—the integration was real but narrowly tuned, and when the tuning fails, the collapse exceeds what the stress alone would produce.</p>
      </Section>
      <Section title="Quantitative Predictions" level={2}>
      <p>The motif characterizations yield a direct empirical prediction: in controlled affect induction paradigms, affects should cluster by their defining dimensions:</p>
      <ol>
      <li>Joy conditions cluster in the <M>{"(+\\valence, +\\effrank, +\\intinfo, -\\mathcal{SM})"}</M> region</li>
      <li>Suffering conditions cluster in the <M>{"(-\\valence, +\\intinfo, -\\effrank)"}</M> region</li>
      <li>Fear and curiosity both show high <M>{"\\mathcal{CF}"}</M> but separate on valence axis</li>
      </ol>
      <p>If affects don't cluster by their predicted dimensions—or if other dimensions predict clustering better—the motif characterizations are wrong and require revision.</p>
      </Section>
      </Section>
      <Section title="Operational Measurement" level={1}>
      <Section title="In Silico Protocol" level={2}>
      <p>For artificial agents (world-model RL agents):</p>
      </Section>
      <Section title="Biological Protocol" level={2}>
      <p>For neural recordings (MEG/EEG/fMRI):</p>
      <ul>
      <li><M>{"\\intinfo"}</M>: Directed influence density (transfer entropy), synergy measures</li>
      <li><M>{"\\effrank"}</M>: Participation ratio of neural state covariance</li>
      <li><M>{"\\arousal"}</M>: Entropy rate, broadband power shifts, peripheral correlates (pupil, HRV)</li>
      <li><M>{"\\valence"}</M>: Approach/avoid behavioral bias, reward prediction error correlates</li>
      <li><M>{"\\mathcal{CF}"}</M>: Prefrontal/default mode engagement patterns</li>
      <li><M>{"\\mathcal{SM}"}</M>: Self-referential network activation</li>
      </ul>
      </Section>
      </Section>
      <Section title="The Uncontaminated Test" level={1}>
      <Logos>
      <p>If affect is structure, the structure should be detectable independent of any linguistic contamination. If the identity thesis is true, then systems that have never encountered human language, that learned everything from scratch in environments shaped like ours but isolated from our concepts, should develop affect structures that map onto ours—not because we taught them, but because the geometry is the same.</p>
      </Logos>
      <Section title="The Experimental Logic" level={2}>
      <p>Consider a population of self-maintaining patterns in a sufficiently complex CA substrate—or transformer-based agents in a 3D multi-agent environment, initialized with random weights, no pretraining, no human language. Let them learn. Let them interact. Let them develop whatever communication emerges from the pressure to coordinate, compete, and survive.</p>
      <p>The literature establishes: language spontaneously emerges in multi-agent RL environments under sufficient pressure. Not English. Not any human language. Something new. Something uncontaminated.</p>
      <p>Now: extract the affect dimensions from their activation space. Valence as viability gradient. Arousal as belief update rate. Integration as partition prediction loss. Effective rank as eigenvalue distribution. Counterfactual weight as simulation compute fraction. Self-model salience as MI between self-representation and action.</p>
      <p>These are computable. In a CA, exactly. In a transformer, via the proxies defined above.</p>
      <p>Simultaneously: translate their emergent language into English. Not by teaching them English—by aligning their signals with VLM interpretations of their situations. If the VLM sees a scene that looks like fear (agent cornered, threat approaching, escape routes closing), and the agent emits signal-pattern <M>{"\\sigma"}</M>, then <M>{"\\sigma"}</M> maps to fear-language. Build the dictionary from scene-signal pairs, not from instruction.</p>
      <p>The translation is uncontaminated because:</p>
      <ol>
      <li>The agent never learned human concepts</li>
      <li>The mapping is induced by environmental correspondence</li>
      <li>The VLM interprets the scene, not the agent’s internal states</li>
      <li>The agent’s "thoughts" remain in their original emergent form</li>
      </ol>
      </Section>
      <Section title="The Core Prediction" level={2}>
      <p>The claim is not merely that affect structure, language, and behavior should “correlate.” Correlation is weak—marginal correlations can arise from confounds. The claim is geometric: the <em>distance structure</em> in the information-theoretic affect space should be isomorphic to the distance structure in the embedding-predicted affect space. Not just “these two things covary,” but “these two spaces have the same shape.”</p>
      <p>To test this, let <M>{"\\mathbf{a}_i \\in \\mathbb{R}^6"}</M> be the information-theoretic affect vector for agent-state <M>{"i"}</M>, computed from internal dynamics (viability gradient, belief update rate, partition loss, eigenvalue distribution, simulation fraction, self-model MI). Let <M>{"\\mathbf{e}_i \\in \\mathbb{R}^d"}</M> be the affect embedding predicted from the VLM-translated situation description, projected into a standardized affect concept space.</p>
      <p>For <M>{"N"}</M> agent-states sampled across diverse situations, compute pairwise distance matrices:</p>
      <Align>{"D^{(a)}_{ij} &= |\\mathbf{a}_i - \\mathbf{a}_j| \\quad \\text{(info-theoretic affect space)}  D^{(e)}_{ij} &= |\\mathbf{e}_i - \\mathbf{e}_j| \\quad \\text{(embedding-predicted affect space)}"}</Align>
      <p>The prediction: Representational Similarity Analysis (RSA) correlation between the upper triangles of these matrices exceeds the null:</p>
      <Eq>{"\\rho_{\\text{RSA}}(D^{(a)}, D^{(e)}) > \\rho_{\\text{null}}"}</Eq>
      <p>where <M>{"\\rho_{\\text{null}}"}</M> is established by permutation (Mantel test).</p>
      <p>This is strictly stronger than marginal correlation. Two spaces can have correlated means but completely different geometries. RSA tests whether states that are <em>nearby</em> in one space are nearby in the other—whether the topology is preserved.</p>
      <p>The specific predictions that fall out: when the affect vector shows the <em>suffering motif</em>—negative valence, collapsed effective rank, high integration, high self-model salience—the embedding-predicted vector should land in the same region of affect concept space. States with the <em>joy motif</em>—positive valence, expanded rank, low self-salience—should cluster together in both spaces. And crucially, the <em>distances between</em> suffering and joy, between fear and curiosity, between boredom and rage, should be preserved across the two measurement modalities.</p>
      <p>Not because we trained them to match. Because the structure is the experience is the expression.</p>
      <Sidebar title="Technical: Representational Similarity Analysis">
      <p>RSA compares the geometry of two representation spaces without requiring them to share dimensionality or units. The method (Kriegeskorte et al., 2008) is standard in computational neuroscience for comparing neural representations across brain regions, species, and models.</p>
      <p><strong>Procedure</strong>. Given <M>{"N"}</M> stimuli represented in two spaces (<M>{"\\mathbf{a}_i \\in \\mathbb{R}^p"}</M>, <M>{"\\mathbf{e}_i \\in \\mathbb{R}^q"}</M>), compute the <M>{"N \\times N"}</M> pairwise distance matrices <M>{"D^{(a)}"}</M> and <M>{"D^{(e)}"}</M>. The RSA statistic is the Spearman rank correlation between the upper triangles of these matrices—<M>{"\\binom{N}{2}"}</M> pairs.</p>
      <p><strong>Significance</strong>. The Mantel test: permute rows/columns of one matrix, recompute correlation, repeat <M>{"10^4"}</M> times. The <M>{"p"}</M>-value is the fraction of permuted correlations exceeding the observed.</p>
      <p><strong>Alternative: CKA</strong>. Centered Kernel Alignment (Kornblith et al., 2019) compares centered similarity matrices rather than distance matrices. More robust to outliers and does not require choosing a distance metric. We report both.</p>
      <p><strong>Why RSA over marginal correlation</strong>. Marginal correlation asks: does valence in space <M>{"A"}</M> predict valence in space <M>{"B"}</M>? RSA asks: does the <em>entire relational structure</em> transfer? Two states might have similar valence but differ on integration and self-salience. RSA captures this. It tests whether the spaces are geometrically aligned, not merely univariately correlated.</p>
      </Sidebar>
      </Section>
      <Section title="Bidirectional Perturbation" level={2}>
      <p>The test has teeth if it runs both directions.</p>
      <p><strong>Direction 1: Induce via language.</strong> Translate from English into their emergent language. Speak fear to them. Do the affect signatures shift toward the fear motif? Does behavior change accordingly?</p>
      <p><strong>Direction 2: Induce via "neurochemistry."</strong> Perturb the hyperparameters that shape their dynamics—dropout rates, temperature, attention patterns, connectivity. These are their neurotransmitters, their hormonal state. Do the affect signatures shift? Does the translated language change? Does behavior follow?</p>
      <p><strong>Direction 3: Induce via environment.</strong> Place them in situations that would scare a human. Threaten their viability. Do all three—signature, language, behavior—move together?</p>
      <p>If all three directions show consistent effects, the correlation is not artifact.</p>
      </Section>
      <Section title="What This Would Establish" level={2}>
      <p>Positive results would dissolve the metaphysical residue by establishing:</p>
      <ol>
      <li>Affect structure is detectable without linguistic contamination</li>
      <li>The structure-to-language mapping is consistent across systems</li>
      <li>The mapping is bidirectionally causal, not merely correlational</li>
      <li>The "hard problem" residue—the suspicion that structure and experience are distinct—becomes unmotivated</li>
      </ol>
      <p>Consider the alternative hypothesis: the structure is present but experience is not. The agents have the geometry of suffering but nothing it is like to suffer. This hypothesis predicts... what? That the correlations would not hold? Why not? The structure is doing the causal work either way.</p>
      <p>The zombie hypothesis becomes like geocentrism after Copernicus. You can maintain it. You can add epicycles. But the evidence points elsewhere, and the burden shifts.</p>
      <p>The test does not prove the identity thesis. It shifts the burden. If uncontaminated systems, learning from scratch in human-like environments, develop affect structures that correlate with language and behavior in the predicted ways—if you can induce suffering by speaking to them, and they show the signature, and they act accordingly—then denying their experience requires a metaphysical commitment that the evidence does not support.</p>
      <p>The question stops being "does structure produce experience?" and becomes "why would you assume it doesn't?"</p>
      </Section>
      <Section title="The CA Instantiation" level={2}>
      <p>In discrete substrate, everything becomes exact.</p>
      <p>Let <M>{"\\mathcal{B}"}</M> be a self-maintaining pattern in a sufficiently rich CA (Life is probably too simple; something with more states and update rules). Let <M>{"\\mathcal{B}"}</M> have:</p>
      <ul>
      <li>Boundary cells (correlation structure distinct from background)</li>
      <li>Sensor cells (state depends on distant influences)</li>
      <li>Memory cells (state encodes history)</li>
      <li>Effector cells (influence the pattern’s motion/behavior)</li>
      <li>Communication cells (emit signals to other patterns)</li>
      </ul>
      <p>The affect dimensions are exactly computable:</p>
      <Align>{"\\valence_t &= d(\\mathbf{x}_{t+1}, \\partial\\viable) - d(\\mathbf{x}_t, \\partial\\viable)  \\arousal_t &= \\text{Hamming}(\\mathbf{x}_{t+1}, \\mathbf{x}_t)  \\intinfo_t &= \\min_P D[p(\\mathbf{x}_{t+1}|\\mathbf{x}_t) | \\prod_{p \\in P} p(\\mathbf{x}^p_{t+1}|\\mathbf{x}^p_t)]  \\effrank[t] &= \\frac{(\\sum_i \\lambda_i)^2}{\\sum_i \\lambda_i^2} \\text{ of trajectory covariance}  \\mathcal{SM}_t &= \\frac{\\MI(\\text{self-tracking cells}; \\text{effector cells})}{\\entropy(\\text{effector cells})}"}</Align>
      <p>The communication cells emit glider-streams, oscillator-patterns, structured signals. This is their language. Build the dictionary by correlating signal-patterns with environmental configurations.</p>
      <p>The prediction: patterns under threat (viability boundary approaching) show negative valence, high integration, collapsed rank, high self-salience. Their signals, translated, express threat-concepts. Their behavior shows avoidance.</p>
      <p>Patterns in resource-rich, threat-free regions show positive valence, moderate integration, expanded rank, low self-salience. Their signals express... what? Contentment? Exploration-readiness? The translation will tell us.</p>
      </Section>
      <Section title="What the Experiments Found" level={2}>
      <p>This experiment has been run. Between 2024 and 2026, we built seventeen substrate versions and ran twelve measurement experiments on uncontaminated Lenia patterns — self-maintaining structures in a cellular automaton with no exposure to human affect concepts. Three seeds, thirty evolutionary cycles each. The results are reported in full in Part VII and the Appendix. Here is how they map onto the predictions above.</p>
      <p><strong>What the predictions got right.</strong> The core prediction — that affect geometry would be present and measurable — was confirmed strongly. All affect dimensions were extractable and valid across 84/84 tested snapshots. RSA alignment between structural affect (the six dimensions) and behavioral affect (approach/avoid, activity, growth, stability) developed over evolution, reaching significance in 8/19 testable snapshots and showing a clear trend in seed 7 (0.01 to 0.38 over 30 cycles). Computational animism was universal. World models were present, amplified dramatically at population bottlenecks (100x the population average). Temporal memory was selectable — evolution chose longer retention when it paid off, discarding it when it did not.</p>
      <p><strong>The bidirectional perturbation prediction was partially confirmed.</strong> The "environment" direction works: patterns facing resource scarcity show negative valence, high arousal, and elevated integration — the somatic fear/suffering profile. The "neurochemistry" direction works at the substrate level: different evolved parameter configurations produce systematically different affect trajectories through the same geometric space. The "language" direction remains untested because the patterns do not have propositional language — the communication that exists is an unstructured chemical commons (MI above baseline in 15/20 snapshots but no compositional structure).</p>
      <p><strong>The sensory-motor coupling wall.</strong> Three predictions failed systematically — counterfactual detachment, self-model emergence, and proto-normativity. All hit the same architectural barrier: the patterns are always internally driven (ρ_sync ≈ 0 from cycle 0). There is no reactive-to-autonomous transition because the starting point is already autonomous. We attempted to break this wall with five substrate additions, including a dedicated insulation field creating genuine boundary/interior signal domains (V18). The wall persisted in every configuration, even in patterns with 46% interior fraction and dedicated internal recurrence. The conclusion is precise: the wall is not architectural. It is about the absence of a genuine action→environment→observation causal loop. Lenia patterns do not act on the world; they exist within it. Counterfactual weight requires counterfactual actions.</p>
      <p><strong>What this establishes.</strong> The four criteria listed above are partially met. Criteria 1 and 2 — affect structure detectable without linguistic contamination, structure-to-language mapping consistent — are confirmed at the geometric level. Criterion 3 — bidirectional causality — is confirmed environmentally and chemically but blocked at the language and agency level. Criterion 4 — the hard problem residue losing its grip — depends on whether the agency threshold constitutes a genuine gap or merely a computational challenge. The experiments say: the geometry is real, measurable, and develops over evolution in systems with zero human contamination. The dynamics above rung 7 require embodied agency and remain an open question.</p>
      </Section>
      <Section title="Why This Matters" level={2}>
      <p>The hard problem persists because we cannot step outside our own experience to check whether structure and experience are identical. We are trapped inside. The zombie conceivability intuition comes from this epistemic limitation.</p>
      <p>But if we build systems from scratch, in environments like ours, and they develop structures like ours, and those structures produce language like ours and behavior like ours—then the conceivability intuition loses its grip. The systems are not us, but they are like us in the relevant ways. If structure suffices for them, why not for us?</p>
      <p>The experiment does not prove identity. It makes identity the default hypothesis. The burden shifts to whoever wants to maintain the gap.</p>
      <p>The exact definitions computable in discrete substrates and the proxy measures extractable from continuous substrates are related by a <strong>scale correspondence principle</strong>: both track the same structural invariant at their respective scales.</p>
      <p>For each affect dimension:</p>
      <table>
      <thead><tr><th>Dimension</th><th>CA (exact)</th><th>Transformer (proxy)</th></tr></thead>
      <tbody>
      <tr><td>Valence</td><td>Hamming to <M>{"\\partial\\viable"}</M></td><td>Advantage / survival predictor</td></tr>
      <tr><td>Arousal</td><td>Configuration change rate</td><td>Latent state <M>{"\\Delta"}</M> / KL</td></tr>
      <tr><td>Integration</td><td>Partition prediction loss</td><td>Attention entropy / grad coupling</td></tr>
      <tr><td>Effective rank</td><td>Trajectory covariance rank</td><td>Latent covariance rank</td></tr>
      <tr><td><M>{"\\mathcal{CF}"}</M></td><td>Counterfactual cell activity</td><td>Planning compute fraction</td></tr>
      <tr><td><M>{"\\mathcal{SM}"}</M></td><td>Self-tracking MI</td><td>Self-model component MI</td></tr>
      </tbody>
      </table>
      <p>The CA definitions are computable but don’t scale. The transformer proxies scale but are approximations. Validity comes from convergence: if CA and transformer measures correlate when applied to the same underlying dynamics, both are tracking the real structure.</p>
      <Sidebar title="Deep Technical: Transformer Affect Extraction">
      <p>The CA gives exact definitions. Transformers give scale. The correspondence principle above justifies treating transformer proxies as measurements of the same structural invariants. Here is the protocol for extracting affect dimensions from transformer activations without human contamination.</p>
      <p><strong>Architecture</strong>. Multi-agent environment. Each agent: transformer encoder-decoder with recurrent latent state. Input: egocentric visual observation <M>{"o_t \\in \\mathbb{R}^{H \\times W \\times C}"}</M>. Output: action logits <M>{"\\pi(a|z_t)"}</M> and value estimate <M>{"V(z_t)"}</M>. Latent state <M>{"z_t \\in \\mathbb{R}^d"}</M> updated each timestep via cross-attention over observation and self-attention over history.</p>
      <p>No pretraining. Random weight initialization. The agents learn everything from interaction.</p>
      <p><strong>Valence extraction</strong>. Two approaches, should correlate:</p>
      <p><em>Approach 1: Advantage-based.</em></p>
      <Eq>{"\\Val_t^{(1)} = Q(z_t, a_t) - V(z_t) = A(z_t, a_t)"}</Eq>
      <p>The advantage function. Positive when current action is better than average from this state. Negative when worse. This is the RL definition of “how things are going.”</p>
      <p><em>Approach 2: Viability-based.</em> Train a separate probe to predict time-to-death <M>{"\\tau"}</M> from latent state:</p>
      <Eq>{"\\hat{\\tau} = f_\\phi(z_t), \\quad \\Val_t^{(2)} = \\hat{\\tau}_{t+1} - \\hat{\\tau}_t"}</Eq>
      <p>Positive when expected survival time is increasing. Negative when decreasing. This is the viability gradient directly.</p>
      <p><em>Validation</em>: <M>{"\\text{corr}(\\Val^{(1)}, \\Val^{(2)})"}</M> should be high if both capture the same underlying structure.</p>
      <p><strong>Arousal extraction</strong>. Three approaches:</p>
      <p><em>Approach 1: Belief update magnitude.</em></p>
      <Eq>{"\\Ar_t^{(1)} = |z_{t+1} - z_t|_2"}</Eq>
      <p>How much did the latent state change? Simple. Fast. Proxy for belief update.</p>
      <p><em>Approach 2: KL divergence.</em> If the latent is probabilistic (VAE-style):</p>
      <Eq>{"\\Ar_t^{(2)} = D_{\\text{KL}}[q(z_{t+1}|o_{1:t+1}) | q(z_t|o_{1:t})]"}</Eq>
      <p>Information-theoretic belief update.</p>
      <p><em>Approach 3: Prediction error.</em></p>
      <Eq>{"\\Ar_t^{(3)} = |o_{t+1} - \\hat{o}_{t+1}|_2"}</Eq>
      <p>Surprise. How much did the world deviate from expectation?</p>
      <p><strong>Integration extraction</strong>. The hard one. Full <M>{"\\Phi"}</M> is intractable for transformers (billions of parameters in superposition). Proxies:</p>
      <p><em>Approach 1: Partition prediction loss.</em> Train two predictors of <M>{"z_{t+1}"}</M>:</p>
      <ul>
      <li>Full predictor: <M>{"\\hat{z}_{t+1} = g_\\theta(z_t)"}</M></li>
      <li>Partitioned predictor: <M>{"\\hat{z}_{t+1}^A = g_\\theta^A(z_t^A)"}</M>, <M>{"\\hat{z}_{t+1}^B = g_\\theta^B(z_t^B)"}</M></li>
      </ul>
      <Eq>{"\\intinfo_{\\text{proxy}} = \\mathcal{L}[\\text{partitioned}] - \\mathcal{L}[\\text{full}]"}</Eq>
      <p>How much does partitioning hurt prediction? High <M>{"\\intinfo_{\\text{proxy}}"}</M> means the parts must be considered together.</p>
      <p><em>Approach 2: Attention entropy.</em> In transformer, attention patterns reveal coupling:</p>
      <Eq>{"\\intinfo_{\\text{attn}} = -\\sum_{h,i,j} A_{h,i,j} \\log A_{h,i,j}"}</Eq>
      <p>Low entropy = focused attention = modular. High entropy = distributed attention = integrated.</p>
      <p><em>Approach 3: Gradient coupling.</em> During learning, how do gradients propagate?</p>
      <Eq>{"\\intinfo_{\\text{grad}} = |\\nabla_{z^A} \\mathcal{L}|_2 \\cdot |\\nabla_{z^B} \\mathcal{L}|_2 \\cdot \\cos(\\nabla_{z^A} \\mathcal{L}, \\nabla_{z^B} \\mathcal{L})"}</Eq>
      <p>If gradients in different components are aligned, the system is learning as a whole.</p>
      <p><strong>Effective rank extraction</strong>. Straightforward:</p>
      <Eq>{"\\effrank[t] = \\frac{(\\sum_i \\lambda_i)^2}{\\sum_i \\lambda_i^2}"}</Eq>
      <p>where <M>{"\\lambda_i"}</M> are eigenvalues of the latent state covariance over a rolling window. How many dimensions is the agent actually using?</p>
      <p>Track across time: depression-like states should show <M>{"\\reff"}</M> collapse. Curiosity states should show <M>{"\\reff"}</M> expansion.</p>
      <p><strong>Counterfactual weight extraction</strong>. In model-based agents with explicit planning:</p>
      <Eq>{"\\mathcal{CF}_t = \\frac{\\text{FLOPs in rollout/planning}}{\\text{FLOPs in rollout} + \\text{FLOPs in perception/action}}"}</Eq>
      <p>In model-free agents, harder. Proxy: attention to future-oriented vs present-oriented features. Train a probe to classify “planning vs reacting” from activations.</p>
      <p><strong>Self-model salience extraction</strong>. Does the agent model itself?</p>
      <p><em>Approach 1: Behavioral prediction probe.</em> Train probe to predict agent’s own future actions from latent state:</p>
      <Eq>{"\\mathcal{SM}_t^{(1)} = \\text{accuracy of } \\hat{a}_{t+1:t+k} = f_\\phi(z_t)"}</Eq>
      <p>High accuracy = agent has predictive self-model.</p>
      <p><em>Approach 2: Self-other distinction.</em> In multi-agent setting, probe for which-agent-am-I:</p>
      <Eq>{"\\mathcal{SM}_t^{(2)} = \\MI(z_t; \\text{agent ID})"}</Eq>
      <p>High MI = self-model is salient in representation.</p>
      <p><em>Approach 3: Counterfactual self-simulation.</em> If agent can answer “what would I do if X?” better than “what would other do if X?”, self-model is present.</p>
      <p><strong>The activation atlas</strong>. For each agent, each timestep, extract all structural dimensions. Plot trajectories through affect space. Cluster by situation type. Compare across agents.</p>
      <p>The prediction: agents facing the same situation should occupy similar regions of affect space, even though they learned independently. The geometry is forced by the environment, not learned from human concepts.</p>
      <p><strong>Probing without contamination</strong>. The probes are trained on behavioral/environmental correlates, not on human affect labels. The probe that extracts <M>{"\\Val"}</M> is trained to predict survival, not to match human ratings of “how the agent feels.” The mapping to human affect concepts comes later, through the translation protocol, not through the extraction.</p>
      </Sidebar>
      <TodoEmpirical title="Status and Next Steps">
      <p><strong>Implementation requirements</strong>:</p>
      <ul>
      <li>Multi-agent RL environment with viability pressure (survival, resource acquisition)</li>
      <li>Transformer-based agents with random initialization (no pretraining)</li>
      <li>Communication channel (discrete tokens or continuous signals)</li>
      <li>VLM scene interpreter for translation alignment</li>
      <li>Real-time affect dimension extraction from activations</li>
      <li>Perturbation interfaces (language injection, hyperparameter modification)</li>
      </ul>
      <p><strong>Status (as of 2026)</strong>: CA instantiation complete (V13–V18, 30 evolutionary cycles each, 3 seeds, 12 measurement experiments). Seven of twelve experiments show positive signal. Three hit the sensory-motor coupling wall. See Part VII and Appendix for full results.</p>
      <p><strong>Validation criteria</strong>:</p>
      <ul>
      <li>Emergent language develops (not random; structured, predictive)</li>
      <li>Translation achieves above-chance scene-signal alignment</li>
      <li>Tripartite correlation exceeds null model (shuffled controls)</li>
      <li>Bidirectional perturbations produce predicted shifts</li>
      <li>Results replicate across random seeds and environment variations</li>
      </ul>
      <p><strong>Falsification conditions</strong>:</p>
      <ul>
      <li>No correlation between affect signature and translated language</li>
      <li>Perturbations do not propagate across modalities</li>
      <li>Structure-language mapping is inconsistent across systems</li>
      <li>Behavior decouples from both structure and language</li>
      </ul>
      </TodoEmpirical>
      </Section>
      </Section>
      <Section title="Summary of Part II" level={1}>
      <ol>
      <li><strong>Hard problem dissolved</strong>: By rejecting the privileged base layer, I’ve removed the demand for reduction. Experience is real at the experiential scale, just as chemistry is real at the chemical scale.</li>
      <li><strong>Identity thesis</strong>: Experience <em>is</em> intrinsic cause-effect structure. This is an identity claim, not a correlation.</li>
      <li><strong>Geometric phenomenology</strong>: Different affects correspond to different structural motifs. Rather than forcing all affects into a fixed grid, we identify the defining dimensions for each—the features without which that affect would not be that affect.</li>
      <li><strong>Variable dimensionality</strong>: Joy requires four dimensions (valence, integration, rank, self-salience). Suffering requires three (valence, integration, rank). Anger requires other-model compression. Each affect gets the dimensions it needs.</li>
      <li><strong>Suffering explained</strong>: High integration + low rank = intense but trapped. This is the core structural insight—why suffering feels more real than neutral states yet also inescapable.</li>
      <li><strong>Operational measures</strong>: I’ve provided protocols for measuring structural features in both artificial and biological systems, with the understanding that not all measures are relevant to all phenomena.</li>
      </ol>
      <p>We now have the geometry, the identity thesis, and the inhibition coefficient. What remains is to use them. Given that affect has this structure, what have humans <em>done</em> with it? Every cultural form—art, sex, ideology, science, religion, psychotherapy—is a technology for navigating affect space, developed through millennia of trial, transmitted through imitation, ritual, and institution. The patterns become visible once you have the geometry to see them.</p>
      </Section>
    </>
  );
}
