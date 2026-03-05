// Shared chapter data — safe for both client and server components.
// This is the single source of truth for chapter metadata.

export interface Chapter {
  slug: string;
  title: string;
  shortTitle: string;
}

export const chapters: Chapter[] = [
  { slug: 'introduction', title: 'Introduction', shortTitle: 'Introduction' },
  { slug: 'part-1', title: 'Part I: Thermodynamic Foundations and the Ladder of Emergence', shortTitle: 'Part I: Foundations' },
  { slug: 'part-2', title: 'Part II: The Identity Thesis and the Geometry of Feeling', shortTitle: 'Part II: Identity Thesis' },
  { slug: 'part-3', title: 'Part III: Signatures of Affect Under the Existential Burden', shortTitle: 'Part III: Affect Signatures' },
  { slug: 'part-4', title: 'Part IV: The Topology of Social Bonds', shortTitle: 'Part IV: Social Bonds' },
  { slug: 'part-5', title: 'Part V: Gods and Superorganisms', shortTitle: 'Part V: Gods' },
  { slug: 'part-6', title: 'Part VI: Transcendence and the Shape of Becoming', shortTitle: 'Part VI: Transcendence' },
  { slug: 'epilogue', title: 'Epilogue', shortTitle: 'Epilogue' },
  { slug: 'appendix-empirical', title: 'Appendix: The Empirical Program', shortTitle: 'Empirical Program' },
  { slug: 'appendix-experiments', title: 'Appendix: Experiment Catalog', shortTitle: 'Experiments' },
];
