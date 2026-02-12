import { neon } from '@neondatabase/serverless';
import { drizzle } from 'drizzle-orm/neon-http';
import { migrate } from 'drizzle-orm/neon-http/migrator';
import { readFileSync } from 'fs';
import { createHash } from 'crypto';
import { join } from 'path';

const url = process.env.POSTGRES_URL;
if (!url) {
  console.error('POSTGRES_URL is not set');
  process.exit(1);
}

const sql = neon(url);
const db = drizzle(sql);

const migrationsFolder = join(__dirname, '..', 'drizzle');

async function needsBootstrap(): Promise<boolean> {
  try {
    const result = await sql`
      SELECT count(*)::int as count FROM "drizzle"."__drizzle_migrations"`;
    return result[0].count === 0;
  } catch {
    // Table or schema doesn't exist
    return true;
  }
}

async function bootstrap() {
  console.log('First run detected — seeding migration tracking table...');

  // Create the schema and table that drizzle migrate() expects
  await sql`CREATE SCHEMA IF NOT EXISTS "drizzle"`;
  await sql`
    CREATE TABLE IF NOT EXISTS "drizzle"."__drizzle_migrations" (
      id SERIAL PRIMARY KEY,
      hash text NOT NULL,
      created_at bigint
    )`;

  // Read journal to get migration list
  const journal = JSON.parse(
    readFileSync(join(migrationsFolder, 'meta', '_journal.json'), 'utf-8')
  );

  for (const entry of journal.entries) {
    const filePath = join(migrationsFolder, `${entry.tag}.sql`);
    const content = readFileSync(filePath, 'utf-8');
    const hash = createHash('sha256').update(content).digest('hex');

    // Check if already seeded (idempotent)
    const existing = await sql`
      SELECT 1 FROM "drizzle"."__drizzle_migrations" WHERE hash = ${hash}`;
    if (existing.length === 0) {
      await sql`
        INSERT INTO "drizzle"."__drizzle_migrations" (hash, created_at)
        VALUES (${hash}, ${entry.when})`;
      console.log(`  Seeded: ${entry.tag}`);
    }
  }

  console.log('Bootstrap complete.');
}

async function main() {
  const shouldBootstrap = await needsBootstrap();
  if (shouldBootstrap) {
    await bootstrap();
  }

  console.log('Running migrations...');
  await migrate(db, { migrationsFolder });
  console.log('Migrations complete.');
}

main().catch((err) => {
  console.error('Migration failed:', err);
  process.exit(1);
});
