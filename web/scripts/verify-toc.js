/**
 * Verifies metadata.json matches content files exactly.
 * Compares section ids, titles, levels, ordering, and parent assignments.
 */
const fs = require('fs');
const path = require('path');

const metadata = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'public', 'metadata.json'), 'utf-8'));
const contentDir = path.join(__dirname, '..', 'content');

function slugify(text) {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '').slice(0, 80);
}

function extractSections(content) {
  const sections = [];
  const re = /<Section\s+title="([^"]+)"\s+level=\{(\d)\}(?:\s+id="([^"]+)")?/g;
  let match;
  let currentParent = null;
  while ((match = re.exec(content)) !== null) {
    const title = match[1];
    const level = parseInt(match[2]);
    const id = match[3] || slugify(title);
    const entry = { level, id, text: title };
    if (level === 1) {
      currentParent = id;
    } else if (currentParent) {
      entry.parentSection = currentParent;
    }
    sections.push(entry);
  }
  return sections;
}

let totalErrors = 0;

for (const chapter of metadata) {
  const filePath = path.join(contentDir, chapter.slug + '.tsx');
  if (!fs.existsSync(filePath)) {
    if (chapter.sections.length > 0) {
      console.log(`ERROR [${chapter.slug}]: No content file but ${chapter.sections.length} sections in metadata`);
      totalErrors++;
    }
    continue;
  }

  const content = fs.readFileSync(filePath, 'utf-8');
  const actual = extractSections(content);
  const meta = chapter.sections;

  // Count comparison
  if (actual.length !== meta.length) {
    console.log(`ERROR [${chapter.slug}]: Content has ${actual.length} sections, metadata has ${meta.length}`);
    totalErrors++;
  }

  // Entry-by-entry comparison
  const maxLen = Math.max(actual.length, meta.length);
  for (let i = 0; i < maxLen; i++) {
    const a = actual[i];
    const m = meta[i];
    const prefix = `  [${chapter.slug}] #${i + 1}`;

    if (!a) {
      console.log(`${prefix}: EXTRA in metadata: L${m.level} "${m.text}" (id: ${m.id})`);
      totalErrors++;
      continue;
    }
    if (!m) {
      console.log(`${prefix}: MISSING from metadata: L${a.level} "${a.text}" (id: ${a.id})`);
      totalErrors++;
      continue;
    }

    if (a.id !== m.id) {
      console.log(`${prefix}: ID mismatch: content="${a.id}" vs metadata="${m.id}"`);
      totalErrors++;
    }
    if (a.text !== m.text) {
      console.log(`${prefix}: Title mismatch: content="${a.text}" vs metadata="${m.text}"`);
      totalErrors++;
    }
    if (a.level !== m.level) {
      console.log(`${prefix}: Level mismatch: content=L${a.level} vs metadata=L${m.level}`);
      totalErrors++;
    }
    if ((a.parentSection || null) !== (m.parentSection || null)) {
      console.log(`${prefix}: Parent mismatch: content="${a.parentSection}" vs metadata="${m.parentSection}"`);
      totalErrors++;
    }
  }
}

if (totalErrors === 0) {
  console.log(`ALL GOOD: metadata.json matches all ${metadata.length} content files (${metadata.reduce((a, c) => a + c.sections.length, 0)} sections)`);
} else {
  console.log(`\n${totalErrors} error(s) found`);
  process.exit(1);
}
