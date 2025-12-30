# Makefile for Inevitability book
# Usage: make [target]

.PHONY: all book clean distclean parts view

# Default target
all: book

# Main book compilation
book: book.pdf

book.pdf: book.tex introduction.tex epilogue.tex part1/chapter.tex part2/chapter.tex part3/chapter.tex part4/chapter.tex part5/chapter.tex
	pdflatex -interaction=nonstopmode book.tex || true
	pdflatex -interaction=nonstopmode book.tex || true
	@test -f book.pdf && echo "Book compiled: book.pdf" || (echo "ERROR: book.pdf not created" && exit 1)

# Draft mode (faster, no images)
draft:
	pdflatex -interaction=nonstopmode "\def\draftmode{}\input{book.tex}"
	@echo "Draft compiled: book.pdf"

# Compile individual parts (standalone)
parts: part1.pdf part2.pdf part3.pdf part4.pdf part5.pdf

part1.pdf: part1/thesis_part1.tex
	cd part1 && pdflatex -interaction=nonstopmode thesis_part1.tex && pdflatex -interaction=nonstopmode thesis_part1.tex
	mv part1/thesis_part1.pdf part1.pdf

part2.pdf: part2/thesis_part2.tex
	cd part2 && pdflatex -interaction=nonstopmode thesis_part2.tex && pdflatex -interaction=nonstopmode thesis_part2.tex
	mv part2/thesis_part2.pdf part2.pdf

part3.pdf: part3/thesis_part3.tex
	cd part3 && pdflatex -interaction=nonstopmode thesis_part3.tex && pdflatex -interaction=nonstopmode thesis_part3.tex
	mv part3/thesis_part3.pdf part3.pdf

part4.pdf: part4/thesis_part4.tex
	cd part4 && pdflatex -interaction=nonstopmode thesis_part4.tex && pdflatex -interaction=nonstopmode thesis_part4.tex
	mv part4/thesis_part4.pdf part4.pdf

part5.pdf: part5/thesis_part5.tex
	cd part5 && pdflatex -interaction=nonstopmode thesis_part5.tex && pdflatex -interaction=nonstopmode thesis_part5.tex
	mv part5/thesis_part5.pdf part5.pdf

# Open the PDF (macOS)
view: book.pdf
	open book.pdf

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.toc *.out *.bbl *.blg *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
	rm -f part*/*.aux part*/*.log part*/*.toc part*/*.out

# Clean everything including PDFs
distclean: clean
	rm -f book.pdf part*.pdf
	rm -f part*/*.pdf

# Help
help:
	@echo "Available targets:"
	@echo "  all      - Build the complete book (default)"
	@echo "  book     - Build book.pdf"
	@echo "  draft    - Build book in draft mode (faster, no images)"
	@echo "  parts    - Build all standalone part PDFs"
	@echo "  view     - Open book.pdf (macOS)"
	@echo "  clean    - Remove auxiliary files"
	@echo "  distclean - Remove all generated files including PDFs"
	@echo "  help     - Show this help"
