#!/usr/bin/env node
import { mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import sharp from 'sharp';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');
const outDir = join(root, 'public', 'icons');
const coverPath = join(root, '..', 'book', 'images', 'ascii-art-cover.png');

mkdirSync(outDir, { recursive: true });

const BG_COLOR = { r: 250, g: 250, b: 248, alpha: 1 }; // #fafaf8 — matches site bg

const sizes = [
  { name: 'favicon-32.png', size: 32 },
  { name: 'apple-touch-icon.png', size: 180 },
  { name: 'icon-192.png', size: 192 },
  { name: 'icon-512.png', size: 512 },
];

async function generate() {
  // Load cover image and flatten transparency onto background color
  const cover = sharp(coverPath).flatten({ background: BG_COLOR });
  const meta = await sharp(coverPath).metadata();
  const { width, height } = meta;

  // Make it square by extending the shorter dimension (width) with padding
  const side = Math.max(width, height);
  const padX = Math.floor((side - width) / 2);
  const padY = Math.floor((side - height) / 2);

  const squareCover = cover.extend({
    top: padY,
    bottom: side - height - padY,
    left: padX,
    right: side - width - padX,
    background: BG_COLOR,
  });

  // Get the square buffer once, then resize for each output
  const squareBuf = await squareCover.png().toBuffer();

  for (const { name, size } of sizes) {
    await sharp(squareBuf).resize(size, size).png().toFile(join(outDir, name));
    console.log(`  ${name} (${size}x${size})`);
  }

  // Maskable icon — needs safe zone (center 80%), so add extra padding
  // Take the square image and add 10% padding on each side
  const maskPad = Math.floor(side * 0.1);
  const maskable = sharp(coverPath)
    .flatten({ background: BG_COLOR })
    .extend({
      top: padY + maskPad,
      bottom: side - height - padY + maskPad,
      left: padX + maskPad,
      right: side - width - padX + maskPad,
      background: BG_COLOR,
    });
  await maskable.resize(512, 512).png().toFile(join(outDir, 'maskable-512.png'));
  console.log('  maskable-512.png (512x512)');

  console.log('Done.');
}

generate().catch((err) => {
  console.error(err);
  process.exit(1);
});
