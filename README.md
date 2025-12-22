# Inevitability

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
