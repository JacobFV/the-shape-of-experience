#!/usr/bin/env node

/**
 * Generate AI illustrations from the registry.
 *
 * Usage:
 *   node web/scripts/generate-illustrations.mjs                    # all pending
 *   node web/scripts/generate-illustrations.mjs shadow-of-transcendence  # specific one
 *   node web/scripts/generate-illustrations.mjs --list             # show registry status
 *   node web/scripts/generate-illustrations.mjs --dry-run          # show prompts only
 *
 * Requires: OPENAI_API_KEY env var
 * Output: web/public/images/illustrations/{id}.png
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const OUTPUT_DIR = path.join(ROOT, 'public', 'images', 'illustrations');

// ---------------------------------------------------------------------------
// Parse TS registry by stripping types and evaluating as JS
// ---------------------------------------------------------------------------

const registryPath = path.join(ROOT, 'content', 'illustrations.ts');
const registrySource = fs.readFileSync(registryPath, 'utf-8');

// Strip TypeScript-only syntax so we can evaluate as plain JS
const jsSource = registrySource
  // Remove type imports and interfaces
  .replace(/import\s*\{[^}]*\}\s*from\s*'[^']*';/g, '')
  .replace(/export\s+interface\s+\w+\s*\{[\s\S]*?\n\}/g, '')
  .replace(/export\s+type\s+\w+\s*=\s*[^;]+;/g, '')
  // Remove all type annotations: `: Type`, `: Type[]`, `: Type | undefined`
  .replace(/:\s*(?:ReactNode|IllustrationEntry|Record<[^>]+>|string|number|boolean)(?:\[\])?(?:\s*\|\s*\w+)*/g, '')
  // Remove function return type annotations: ): ReturnType {
  .replace(/\)\s*:\s*\w+(?:\[\])?\s*(?:\|\s*\w+)?\s*\{/g, ') {')
  // Convert exports to plain declarations
  .replace(/export const /g, 'const ')
  .replace(/export function /g, 'function ')
  // Remove 'as const' casts
  .replace(/as\s+const/g, '');

// Evaluate in a function scope to capture variables
const evalFn = new Function(`
  ${jsSource}
  return { ILLUSTRATIONS, HOUSE_STYLE, buildPrompt, getIllustration };
`);

let ILLUSTRATIONS, HOUSE_STYLE, buildPrompt;
try {
  const result = evalFn();
  ILLUSTRATIONS = result.ILLUSTRATIONS;
  HOUSE_STYLE = result.HOUSE_STYLE;
  buildPrompt = result.buildPrompt;
} catch (err) {
  console.error('Failed to parse illustrations.ts:', err.message);
  console.error('Debug: writing stripped JS to /tmp/illustrations-debug.js');
  fs.writeFileSync('/tmp/illustrations-debug.js', jsSource);
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Image generation
// ---------------------------------------------------------------------------

async function generateImage(entry) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    console.error('OPENAI_API_KEY not set');
    process.exit(1);
  }

  const fullPrompt = buildPrompt(entry);

  console.log(`\n  Generating: ${entry.id}`);
  console.log(`  Chapter:    ${entry.chapter}`);
  console.log(`  Size:       ${entry.size || '1024x1024'}`);
  console.log(`  Prompt:     ${fullPrompt.slice(0, 120)}...`);

  const response = await fetch('https://api.openai.com/v1/images/generations', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'dall-e-3',
      prompt: fullPrompt,
      n: 1,
      size: entry.size || '1024x1024',
      quality: 'hd',
      response_format: 'b64_json',
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`OpenAI API error (${response.status}): ${err}`);
  }

  const data = await response.json();
  const b64 = data.data[0].b64_json;
  const revised = data.data[0].revised_prompt;

  const outputPath = path.join(OUTPUT_DIR, `${entry.id}.png`);
  fs.writeFileSync(outputPath, Buffer.from(b64, 'base64'));

  console.log(`  Saved:      ${outputPath}`);
  if (revised) {
    console.log(`  Revised:    ${revised.slice(0, 120)}...`);
  }

  return outputPath;
}

// ---------------------------------------------------------------------------
// Update registry status in the TS source file
// ---------------------------------------------------------------------------

function updateStatus(id, newStatus) {
  let source = fs.readFileSync(registryPath, 'utf-8');

  // Find the entry and update its status
  const pattern = new RegExp(
    `(id:\\s*'${id}'[\\s\\S]*?status:\\s*')\\w+(')`
  );
  if (pattern.test(source)) {
    source = source.replace(pattern, `$1${newStatus}$2`);
    fs.writeFileSync(registryPath, source, 'utf-8');
    console.log(`  Status:     ${id} → ${newStatus}`);
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const args = process.argv.slice(2);

  // --list: show registry
  if (args.includes('--list')) {
    console.log('\nIllustration Registry:');
    console.log('─'.repeat(80));
    for (const e of ILLUSTRATIONS) {
      const icon =
        e.status === 'approved' ? '✅' : e.status === 'generated' ? '🖼️' : '⏳';
      console.log(`  ${icon}  ${e.id.padEnd(30)} ${e.chapter.padEnd(15)} ${e.status}`);
    }
    const pending = ILLUSTRATIONS.filter((e) => e.status === 'pending').length;
    const generated = ILLUSTRATIONS.filter((e) => e.status === 'generated').length;
    const approved = ILLUSTRATIONS.filter((e) => e.status === 'approved').length;
    console.log(`\n  Total: ${ILLUSTRATIONS.length} (${pending} pending, ${generated} generated, ${approved} approved)`);
    return;
  }

  // --dry-run: show prompts
  if (args.includes('--dry-run')) {
    const targets = args.filter((a) => !a.startsWith('--'));
    const toGenerate =
      targets.length > 0
        ? ILLUSTRATIONS.filter((e) => targets.includes(e.id))
        : ILLUSTRATIONS.filter((e) => e.status === 'pending');

    console.log(`\nDry run — ${toGenerate.length} illustration(s):\n`);
    for (const e of toGenerate) {
      const fullPrompt = buildPrompt(e);
      console.log(`── ${e.id} (${e.chapter}) ──`);
      console.log(fullPrompt);
      console.log();
    }
    return;
  }

  // Ensure output directory exists
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  // Specific ID(s) or all pending
  const targets = args.filter((a) => !a.startsWith('--'));
  const toGenerate =
    targets.length > 0
      ? ILLUSTRATIONS.filter((e) => targets.includes(e.id))
      : ILLUSTRATIONS.filter((e) => e.status === 'pending');

  if (toGenerate.length === 0) {
    console.log('Nothing to generate. All illustrations are generated or approved.');
    return;
  }

  console.log(`\nGenerating ${toGenerate.length} illustration(s)...\n`);

  let success = 0;
  let failed = 0;

  for (const entry of toGenerate) {
    try {
      await generateImage(entry);
      updateStatus(entry.id, 'generated');
      success++;
    } catch (err) {
      console.error(`  ERROR: ${entry.id}: ${err.message}`);
      failed++;
    }

    // Rate limit: 1 request per 15 seconds for DALL-E 3
    if (toGenerate.indexOf(entry) < toGenerate.length - 1) {
      console.log('  Waiting 15s (rate limit)...');
      await new Promise((r) => setTimeout(r, 15000));
    }
  }

  console.log(`\nDone: ${success} generated, ${failed} failed.`);
}

main().catch(console.error);
