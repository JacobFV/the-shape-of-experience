-- Add nearestHeadingText column to annotations
ALTER TABLE "annotations" ADD COLUMN "nearest_heading_text" text DEFAULT '';
--> statement-breakpoint

-- Change exact from NOT NULL to nullable (bookmarks have empty exact)
ALTER TABLE "annotations" ALTER COLUMN "exact" SET DEFAULT '';
ALTER TABLE "annotations" ALTER COLUMN "exact" DROP NOT NULL;
--> statement-breakpoint

-- Migrate existing bookmarks into annotations
INSERT INTO "annotations" ("user_id", "slug", "nearest_heading_id", "nearest_heading_text", "prefix", "exact", "suffix", "note", "is_published", "created_at", "updated_at")
SELECT "user_id", "slug", "nearest_heading_id", "nearest_heading_text", '', '', '', '', false, "created_at", "created_at"
FROM "bookmarks";
--> statement-breakpoint

-- Drop bookmarks table
ALTER TABLE "bookmarks" DROP CONSTRAINT "bookmarks_user_id_users_id_fk";
--> statement-breakpoint
DROP TABLE "bookmarks";
