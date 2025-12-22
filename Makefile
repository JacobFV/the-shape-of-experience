LATEX = /Library/TeX/texbin/pdflatex
FLAGS = -interaction=nonstopmode

all: part1 part2 part3

part1:
	cd paper/part1 && $(LATEX) $(FLAGS) thesis_part1.tex || true
	cd paper/part1 && $(LATEX) $(FLAGS) thesis_part1.tex || true
	cd paper/part1 && $(LATEX) $(FLAGS) thesis_part1.tex || true

part2:
	cd paper/part2 && $(LATEX) $(FLAGS) thesis_part2.tex || true
	cd paper/part2 && $(LATEX) $(FLAGS) thesis_part2.tex || true
	cd paper/part2 && $(LATEX) $(FLAGS) thesis_part2.tex || true

part3:
	cd paper/part3 && $(LATEX) $(FLAGS) thesis_part3.tex || true
	cd paper/part3 && $(LATEX) $(FLAGS) thesis_part3.tex || true
	cd paper/part3 && $(LATEX) $(FLAGS) thesis_part3.tex || true

clean:
	rm -f paper/*/*.aux paper/*/*.log paper/*/*.out paper/*/*.toc
	rm -f paper/*/*.fls paper/*/*.fdb_latexmk paper/*/*.synctex.gz

cleanall: clean
	rm -f paper/*/*.pdf

.PHONY: all part1 part2 part3 clean cleanall
