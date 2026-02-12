-- Comments on published annotations
CREATE TABLE IF NOT EXISTS "comments" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
  "annotation_id" uuid NOT NULL REFERENCES "annotations"("id") ON DELETE CASCADE,
  "user_id" uuid NOT NULL REFERENCES "users"("id") ON DELETE CASCADE,
  "content" text NOT NULL,
  "created_at" timestamp DEFAULT now()
);

CREATE INDEX IF NOT EXISTS "comments_annotation_id_idx" ON "comments" ("annotation_id");
CREATE INDEX IF NOT EXISTS "comments_user_id_idx" ON "comments" ("user_id");

-- Reactions on annotations or comments
CREATE TABLE IF NOT EXISTS "reactions" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
  "user_id" uuid NOT NULL REFERENCES "users"("id") ON DELETE CASCADE,
  "target_type" text NOT NULL,
  "target_id" uuid NOT NULL,
  "emoji" text NOT NULL,
  "created_at" timestamp DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS "reactions_unique_idx" ON "reactions" ("user_id", "target_type", "target_id", "emoji");
CREATE INDEX IF NOT EXISTS "reactions_target_idx" ON "reactions" ("target_type", "target_id");

-- Purge bookmark rows (annotations with empty exact text)
DELETE FROM "annotations" WHERE "exact" = '' OR "exact" IS NULL;
