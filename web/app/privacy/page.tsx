export default function PrivacyPage() {
  return (
    <div className="app-page" style={{ maxWidth: 640 }}>
      <h1>Privacy Policy</h1>
      <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem' }}>Last updated: February 10, 2026</p>

      <h2>1. What We Collect</h2>
      <p>
        When you create an account, we store your name, email address, and (if using email/password)
        a securely hashed password. If you sign in with GitHub or Google, we receive your name,
        email, and profile image from those services.
      </p>

      <h2>2. How We Use It</h2>
      <p>
        Your account information is used solely to provide the annotation features:
        highlights, bookmarks, notes, and community notes. We do not sell your data
        or use it for advertising.
      </p>

      <h2>3. Community Notes</h2>
      <p>
        Notes you publish are visible to other users alongside your display name and profile image.
        Private annotations (highlights, bookmarks, unpublished notes) are visible only to you.
      </p>

      <h2>4. Third-Party Services</h2>
      <p>
        We use Vercel for hosting, Neon for database hosting, and GitHub/Google for optional
        OAuth authentication. Each service has its own privacy policy governing data they process.
      </p>

      <h2>5. Cookies</h2>
      <p>
        We use a session cookie to keep you signed in. We do not use tracking cookies
        or third-party analytics.
      </p>

      <h2>6. Data Deletion</h2>
      <p>
        You can delete your account and all associated data by contacting Jacob Valdez
        through the links in the sidebar.
      </p>

      <h2>7. Changes</h2>
      <p>
        We may update this policy as the site evolves. Material changes will be noted
        by updating the date above.
      </p>
    </div>
  );
}
