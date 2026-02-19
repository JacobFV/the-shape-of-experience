# The Shape of Experience

> **Work in Progress** — This is an active research project, not a finished publication. Content is incomplete, speculative, and subject to change. Claims should be treated as hypotheses under investigation, not established conclusions.

A geometric theory of affect for biological and artificial systems.

**Read online**: [theshapeofexperience.com](https://theshapeofexperience.com)

## Repository Structure

This repository is a **research center** — the canonical source of truth for the project. Everything downstream (web book, PDF, audiobook, videos, talks, papers) is generated from here.

- `web/` — Next.js 15 web book (Vercel deployment)
- `web/content/` — Chapter content (TSX components)
- `empirical/experiments/` — Experiment code and results (V2–V24)
- `book/` — LaTeX source (secondary, compiled to PDF)
- `EXPERIMENTS.md` — Experiment catalog and research roadmap
- `CLAUDE.md` — Research context and AI collaboration instructions

## Status

This is research in early stages. The empirical program (V2–V24) is actively running. The theoretical framework is under development. Many claims are speculative and await experimental validation. See `EXPERIMENTS.md` for what has been tested and what remains.

## LaTeX Setup (macOS)

### 1. Install BasicTeX

```bash
brew install --cask basictex
```

Restart your terminal or run:
```bash
eval "$(/usr/libexec/path_helper)"
```

### 2. Configure Package Manager

```bash
sudo /Library/TeX/texbin/tlmgr update --self
sudo /Library/TeX/texbin/tlmgr install latexmk texliveonfly
```

### 3. Compile Documents

Use `texliveonfly` to auto-install missing packages:
```bash
texliveonfly paper/your_document.tex
```

Or install packages manually:
```bash
sudo /Library/TeX/texbin/tlmgr install <package-name>
```

### 4. VS Code Setup (Optional)

Install the **LaTeX Workshop** extension, then add to your settings.json:

```json
{
  "latex-workshop.latex.tools": [
    {
      "name": "texliveonfly",
      "command": "texliveonfly",
      "args": [
        "--compiler=pdflatex",
        "--arguments=-synctex=1 -interaction=nonstopmode -file-line-error",
        "%DOC%"
      ]
    }
  ],
  "latex-workshop.latex.recipes": [
    {
      "name": "texliveonfly",
      "tools": ["texliveonfly"]
    }
  ]
}
```
