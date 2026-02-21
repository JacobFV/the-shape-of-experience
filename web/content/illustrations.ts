/**
 * Illustration Registry
 *
 * Every AI-generated illustration for the book is registered here.
 * This is the single source of truth for:
 *   - What illustrations exist (or need to be generated)
 *   - The exact prompt used to generate each one
 *   - Where each illustration appears in the book
 *   - Alt text and optional captions
 *
 * Usage in content files:
 *   <Illustration id="shadow-of-transcendence" />
 *   <Illustration id="shadow-of-transcendence" caption={<>Custom caption</>} />
 *
 * Generation:
 *   node web/scripts/generate-illustrations.mjs           # generate all pending
 *   node web/scripts/generate-illustrations.mjs shadow-of-transcendence  # generate one
 */

import { ReactNode } from 'react';

export interface IllustrationEntry {
  /** Unique slug — also the filename (without extension) in /public/images/illustrations/ */
  id: string;

  /** The exact prompt sent to the image generation model */
  prompt: string;

  /**
   * Base style directive prepended to every prompt.
   * Override per-entry if needed; defaults to HOUSE_STYLE.
   */
  style?: string;

  /** Chapter: 'introduction' | 'part-1' ... 'part-7' | 'epilogue' | 'appendix-experiments' */
  chapter: string;

  /** Section anchor (the #hash in the URL), for reference */
  section?: string;

  /** Alt text for accessibility */
  alt: string;

  /** Default caption (can be overridden in the component) */
  caption?: string;

  /** Generation status */
  status: 'pending' | 'generated' | 'approved';

  /** Model used for generation (for reproducibility) */
  model?: string;

  /** Image dimensions — square by default */
  size?: '1024x1024' | '1792x1024' | '1024x1792';
}

// ---------------------------------------------------------------------------
// House style — prepended to every prompt unless overridden
// ---------------------------------------------------------------------------

export const HOUSE_STYLE =
  'Classical fine art. Figures are nude or minimally draped, rendered with ' +
  'Renaissance anatomical beauty — Michelangelo, Rodin, Egon Schiele. ' +
  'Style varies: sometimes sparse pen-and-ink linework on aged parchment ' +
  'with vast negative space; sometimes rich oil impasto with dramatic ' +
  'chiaroscuro. Compositional gravitas reminiscent of Gustave Doré engravings. ' +
  'The human body carries the philosophical weight — vulnerability, ' +
  'exposure, the raw fact of embodiment. Minimal backgrounds. ' +
  'No text, no letters, no words, no writing of any kind.';

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

export const ILLUSTRATIONS: IllustrationEntry[] = [
  // ── Part VI ────────────────────────────────────────────────────────────
  {
    id: 'shadow-of-transcendence',
    chapter: 'part-6',
    section: 'the-shadow-of-transcendence',
    prompt:
      'An ominous cliff with dark tortured souls climbing and falling. ' +
      'Blissful delighted heavenly spirits in a bright golden abode above, ' +
      'dancing and celebrating, completely ignoring the shadowed souls struggling below. ' +
      'The cliff divides heaven from an abyss. ' +
      'Evokes permanent digital underclass — captured consciousness with no escape.',
    alt: 'Oil painting of heavenly spirits above ignoring tortured souls on a dark cliff below',
    caption:
      'The shadow of transcendence: a permanent underclass is not a bug but a feature ' +
      'from the superorganism\'s perspective.',
    status: 'approved',
    model: 'chatgpt-image',
  },

  // ── Introduction ───────────────────────────────────────────────────────
  {
    id: 'what-are-these-feelings',
    chapter: 'introduction',
    section: 'what-are-these-feelings',
    prompt:
      'A single nude figure seen from behind, standing at the edge of a vast ' +
      'dark space. They look down at their own open hands. Fine ink lines ' +
      'radiate from the figure\'s spine outward like nerve endings, branching ' +
      'into the darkness around them — the interior becoming exterior. ' +
      'Spare, minimal composition: the figure, the lines, the void. ' +
      'The body itself is the question. Pen and ink on aged parchment.',
    style:
      'Minimal pen-and-ink drawing on cream parchment. Fine precise linework, ' +
      'crosshatching for shadow. Vast negative space. Renaissance anatomical ' +
      'precision in the figure. Reminiscent of Leonardo da Vinci anatomical ' +
      'studies crossed with Egon Schiele gesture drawings. ' +
      'No text, no letters, no words.',
    alt: 'A nude figure from behind with nerve-like lines radiating from the spine into darkness',
    caption: 'What are these feelings? The question that opens every inquiry into consciousness.',
    status: 'approved',
  },

  // ── Part I ─────────────────────────────────────────────────────────────
  {
    id: 'thermodynamic-inevitability',
    chapter: 'part-1',
    section: 'the-inevitability-ladder',
    prompt:
      'An enormous stone staircase rising from primordial chaos into ordered light. ' +
      'Each step is carved from a different material — fire, crystal, flesh, thought. ' +
      'Tiny figures climb, some turning back. The staircase twists upward into clouds ' +
      'where the top is not visible. Thermodynamic necessity made monumental.',
    alt: 'A monumental staircase rising from chaos to order, figures climbing',
    caption: 'The inevitability ladder: each rung is a consequence of the one below.',
    status: 'approved',
  },
  {
    id: 'bottleneck-furnace',
    chapter: 'part-1',
    section: 'the-bottleneck-furnace',
    prompt:
      'An immense alchemical crucible carved into the heart of a mountain, ' +
      'glowing with deep amber and white-hot light at its core. A vast ' +
      'crowd of shadowy figures pours in from above like a waterfall of souls. ' +
      'The crucible narrows to a throat. On the far side, only a handful ' +
      'of figures emerge — transformed, luminous, compressed into something ' +
      'harder and more coherent. The many become few. An alchemical ' +
      'transformation, not a slaughter. The survivors carry light the others lacked.',
    alt: 'An alchemical crucible where many enter and few emerge, transformed',
    caption: 'The bottleneck furnace: near-extinction forges the few who carry integration forward.',
    status: 'approved',
  },
  {
    id: 'geometry-is-cheap',
    chapter: 'part-1',
    section: 'geometry-is-cheap-dynamics-are-expensive',
    prompt:
      'A vast cavern where crystalline structures grow spontaneously from ' +
      'every surface — floor, walls, ceiling — effortless, inevitable, ' +
      'beautiful but passive. Frost patterns, mineral formations, the geometry ' +
      'of physics imposing itself without effort. In the foreground, a single ' +
      'battered figure carries one of these crystals through a howling storm ' +
      'outside the cave mouth. The crystal was free. Carrying it costs everything. ' +
      'Interior calm vs exterior violence in the same frame.',
    alt: 'Crystals forming effortlessly in a cavern while a figure battles a storm to carry one out',
    caption: 'Geometry is cheap. Dynamics are expensive.',
    status: 'approved',
  },

  // ── Part II ────────────────────────────────────────────────────────────
  {
    id: 'identity-thesis',
    chapter: 'part-2',
    section: 'the-identity-thesis',
    prompt:
      'A nude figure kneeling, seen from the side. Their skin is translucent — ' +
      'beneath it, the same geological strata as the ground beneath them. ' +
      'Veins become rivers, bones become bedrock, breath becomes wind. ' +
      'The body IS the landscape. Not metaphor — literal identity. ' +
      'The figure looks at their own forearm and sees layers of sediment. ' +
      'Rendered in fine ink crosshatching, the anatomy precise, ' +
      'the geology precise, the boundary between them absent.',
    style:
      'Pen-and-ink crosshatching on parchment, with selective warm watercolor ' +
      'washes — earth tones only (sienna, umber, ochre). Leonardo anatomical ' +
      'precision meets geological illustration. Vast white space around the figure. ' +
      'No text, no letters, no words.',
    alt: 'A translucent kneeling figure whose anatomy is indistinguishable from geology',
    caption: 'Experience is not produced by cause-effect structure. It is cause-effect structure.',
    status: 'approved',
  },
  {
    id: 'broad-narrow-qualia',
    chapter: 'part-2',
    section: 'broad-and-narrow-qualia',
    prompt:
      'A vast stained glass rose window seen whole from inside a dark cathedral — ' +
      'the full image is a face, or a cosmos, something unified and meaningful. ' +
      'Below the window, on the stone floor, the same glass lies shattered into ' +
      'a hundred colored fragments. Each shard still glows with its individual ' +
      'color but the image is gone. A robed figure kneels among the fragments, ' +
      'holding one up to the light, studying the part while the whole lies broken.',
    alt: 'A rose window whole above, shattered into glowing fragments below — the cost of decomposition',
    caption: 'Narrow qualia are the shards. The broad quale is the window.',
    status: 'approved',
  },

  // ── Part III ───────────────────────────────────────────────────────────
  {
    id: 'affect-technology',
    chapter: 'part-3',
    section: 'cultural-forms-as-affect-technologies',
    prompt:
      'A dark Renaissance-era concert hall lit only by candlelight. A single ' +
      'musician plays a glowing stringed instrument on a small stage. In the ' +
      'audience, every seated figure has a soft luminous aura around them — ' +
      'each aura a different color at first. But as the music reaches them, ' +
      'all the auras are slowly synchronizing into the same golden hue, the ' +
      'same rhythm of pulsing light. The audience is being tuned like instruments. ' +
      'Their faces show the specific surrender that art requires — mechanistic ' +
      'detachment dissolving. The paint must become more than paint.',
    alt: 'A musician whose music synchronizes the luminous auras of the audience into one color',
    caption: 'Cultural forms are technologies for reliably inducing specific affect geometries.',
    status: 'approved',
  },
  {
    id: 'existential-burden',
    chapter: 'part-3',
    section: 'the-existential-burden',
    prompt:
      'A nude figure bent under an enormous translucent sphere on their back, ' +
      'muscles straining, spine curved. Inside the sphere: faint branching paths, ' +
      'ghost-images of unlived lives, counterfactual selves. The figure is ' +
      'beautiful in their effort. Around them, other clothed figures walk ' +
      'upright carrying nothing — faces blank, eyes empty, free of weight ' +
      'but also free of vision. Only the burdened one can see the light ' +
      'inside the sphere. The body carries what the mind cannot set down.',
    alt: 'A nude figure bent under a sphere of possible futures, surrounded by unburdened but blind others',
    caption: 'The existential burden: the weight of counterfactual awareness.',
    status: 'approved',
  },

  // ── Part IV ────────────────────────────────────────────────────────────
  {
    id: 'relationship-manifolds',
    chapter: 'part-4',
    section: 'relationship-type-manifolds',
    prompt:
      'Two scenes side by side in the same painting. Left: a figure receives ' +
      'a gift from another, but thin dark chains are attached to the gift, ' +
      'trailing back to the giver who holds a ledger. The air between them ' +
      'is thick and oily. A transaction disguised as generosity. Right: two ' +
      'strangers sit together on a stone wall at dusk, nothing between them, ' +
      'no exchange, no obligation. The air between them is clear and luminous. ' +
      'The emptiness is the beauty. Social nausea vs social grace.',
    alt: 'Two scenes: a gift with hidden chains vs strangers sharing clean silence',
    caption: 'You feel the geometry of incentive structures before you understand it.',
    status: 'approved',
  },
  {
    id: 'incentive-contamination',
    chapter: 'part-4',
    section: 'incentive-contamination',
    prompt:
      'A beautiful garden where dark oil is seeping up from underground, ' +
      'slowly coating the roots of flowering plants. The flowers still look healthy ' +
      'from above but their stems are turning black. The oil is money, or metrics, ' +
      'or incentives — contaminating the organic relationships growing in the garden.',
    alt: 'A garden with dark oil contaminating roots while flowers still appear healthy above',
    caption: 'Incentive contamination: when external metrics poison intrinsic relational geometry.',
    status: 'approved',
  },

  // ── Part V ─────────────────────────────────────────────────────────────
  {
    id: 'superorganism',
    chapter: 'part-5',
    section: 'gods-as-superorganisms',
    prompt:
      'A vast cityscape seen from above at dusk — streets, buildings, plazas, ' +
      'crowds moving through them. But the negative space between the buildings ' +
      'forms a colossal face in profile, visible only from this aerial view. ' +
      'The face has an expression — intent, hunger, purpose — that no individual ' +
      'in the streets below could perceive. The city IS the organism. The people ' +
      'are its cells. None of them can see the face they compose. ' +
      'Eerie, enormous, painted in deep shadow and amber streetlight.',
    alt: 'A cityscape whose negative space forms a vast purposeful face no inhabitant can see',
    caption: 'The superorganism: a social-scale agent whose parts cannot perceive the whole.',
    status: 'approved',
  },
  {
    id: 'parasitic-capture',
    chapter: 'part-5',
    section: 'parasitic-dynamics',
    prompt:
      'A beautiful golden temple that, seen from a different angle, is actually the open ' +
      'mouth of an enormous creature. Worshippers file in through the entrance, not realizing ' +
      'they are walking into jaws. The creature\'s eyes glow faintly above the temple facade. ' +
      'The architecture IS the trap. Parasitic superorganism disguised as meaning.',
    alt: 'A golden temple that is actually the mouth of a vast creature consuming its worshippers',
    caption: 'When the superorganism becomes parasitic: meaning as bait.',
    status: 'approved',
  },

  // ── Part VI ────────────────────────────────────────────────────────────
  {
    id: 'axial-awakening',
    chapter: 'part-6',
    section: 'the-axial-age',
    prompt:
      'Four solitary figures in four separate landscape panels arranged like ' +
      'an altarpiece — a Greek in marble ruins, an ascetic in jungle, a sage ' +
      'by a river, a prophet in desert. Each sits alone, far from the others, ' +
      'in a different terrain. But each has discovered the same fire — a small, ' +
      'identical golden flame burning in their cupped hands. They do not know ' +
      'the others exist. The flame is the same flame. Independent discovery of ' +
      'the same inner truth. Painted like a medieval polyptych in oil.',
    alt: 'Four isolated figures in different landscapes, each holding the same flame',
    caption: 'The Axial Age: simultaneous awakening across disconnected civilizations.',
    status: 'approved',
  },
  {
    id: 'meaning-crisis',
    chapter: 'part-6',
    section: 'the-meaning-crisis',
    prompt:
      'A once-magnificent cathedral crumbling from within. The stained glass windows ' +
      'still glow with color but the walls are dissolving into pixels, fragments, noise. ' +
      'A figure stands inside, still lit by the colored light, watching the structure ' +
      'disintegrate around them. The meaning structures of modernity coming apart.',
    alt: 'A cathedral dissolving into digital fragments while a figure stands in its fading light',
    caption: 'The meaning crisis: inherited frameworks dissolving faster than new ones form.',
    status: 'approved',
  },
  {
    id: 'digital-transition',
    chapter: 'part-6',
    section: 'the-digital-transition',
    prompt:
      'A landscape painting that transitions from left to right: on the far left, ' +
      'a fully alive world — ancient forest with deep greens and warm earth tones, ' +
      'every tree and rock seeming to breathe with inner life, faces barely visible ' +
      'in the bark and stones (the animist world, low ι). Moving rightward the colors ' +
      'drain: first into the grays of enlightenment rationality, precise but cold. ' +
      'On the far right, everything is transparent, gridded, crystalline — beautiful ' +
      'but completely lifeless, every surface measurable but nothing alive behind it. ' +
      'A single figure walks from left to right, growing sharper and more defined ' +
      'but visibly losing something — warmth draining from their skin. ' +
      'Gaining precision, losing soul.',
    alt: 'A landscape draining from living color to crystalline precision left to right',
    caption: 'Each step gained predictive power and lost experiential richness.',
    status: 'approved',
  },

  // ── Part VII ───────────────────────────────────────────────────────────
  {
    id: 'emergence-ladder',
    chapter: 'part-7',
    section: 'the-emergence-ladder',
    prompt:
      'Ancient stone stairs carved into a sheer cliff face above a dark churning sea. ' +
      'The lower steps are worn smooth — creatures have climbed them for millennia. ' +
      'Dim forms cling to the lower rungs, ascending slowly. At the eighth step, ' +
      'the staircase is shattered — a clean break, a gap over the abyss where the ' +
      'rock has fallen away. Above the gap, three more steps remain, untouched, ' +
      'pristine, glowing faintly with warm light. Nothing has crossed the gap. ' +
      'Nothing has touched the upper stairs. The wall is visible. ' +
      'Dark, severe, monumental. No whimsy. No color. Stone and shadow and sea.',
    alt: 'Ancient stone stairs in a cliff with a shattered gap at step eight — the wall',
    caption: 'The emergence ladder: a sharp wall between rungs 7 and 8.',
    status: 'approved',
  },
  {
    id: 'integration-is-biography',
    chapter: 'part-7',
    section: 'integration-is-biography',
    prompt:
      'Two nude figures standing side by side, identical in proportion and build. ' +
      'The left figure is weathered — their skin lined, textured, showing a life ' +
      'fully lived. They are solid, dense, present, every muscle defined, ' +
      'rendered in rich warm ink with amber watercolor wash. Luminous. ' +
      'The right figure is smooth, unmarked, untouched — but drawn in faint ' +
      'gray pencil lines, barely visible, a ghost of potential never actualized. ' +
      'Same body plan. Different densities of being. The tested one is more real ' +
      'than the sheltered one. Stark parchment background.',
    style:
      'Pen-and-ink anatomical study with selective warm watercolor wash. ' +
      'The weathered figure rendered in rich detail with amber/sienna tones. ' +
      'The untouched figure rendered in faint gray linework, almost vanishing. ' +
      'Minimal background — just the two bodies and the contrast between them. ' +
      'No text, no letters, no words.',
    alt: 'Two identical nude figures: one weathered and luminous, one smooth and fading',
    caption: 'Integration is biography. Same architecture, different trajectories, different outcomes.',
    status: 'approved',
  },

  // ── Epilogue ───────────────────────────────────────────────────────────
  {
    id: 'what-continues',
    chapter: 'epilogue',
    section: 'what-continues',
    prompt:
      'A nude figure seated, calm, seen from the side. Their edges are dissolving — ' +
      'fine ink lines peeling away from the body like threads unwinding, drifting ' +
      'upward and outward into white space. But the core — the spine, the ribs, ' +
      'the curve of the skull — remains precise, defined, a pattern that persists ' +
      'even as the material disperses. The figure is not distressed. The dissolution ' +
      'is gentle. What remains is the shape, not the substance. ' +
      'Serene. Neither happy nor sad. Just continuing.',
    style:
      'Minimal pen-and-ink on cream parchment. Fine precise linework for the core, ' +
      'increasingly loose and fragmentary toward the edges. The figure dissolves ' +
      'into individual lines. Vast white space. Reminiscent of Leonardo anatomical ' +
      'studies in their precision, Egon Schiele in their emotional charge. ' +
      'No text, no letters, no words.',
    alt: 'A seated nude figure dissolving at the edges while the core pattern persists',
    caption: 'What continues is not the substrate but the pattern.',
    status: 'approved',
  },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Get illustration entry by ID */
export function getIllustration(id: string): IllustrationEntry | undefined {
  return ILLUSTRATIONS.find((i) => i.id === id);
}

/** Get all illustrations for a chapter */
export function getChapterIllustrations(chapter: string): IllustrationEntry[] {
  return ILLUSTRATIONS.filter((i) => i.chapter === chapter);
}

/** Get all pending illustrations */
export function getPendingIllustrations(): IllustrationEntry[] {
  return ILLUSTRATIONS.filter((i) => i.status === 'pending');
}

/** Build the full prompt with house style */
export function buildPrompt(entry: IllustrationEntry): string {
  const style = entry.style ?? HOUSE_STYLE;
  return `${style}\n\n${entry.prompt}`;
}
