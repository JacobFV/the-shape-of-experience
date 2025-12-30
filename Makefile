LATEX = /Library/TeX/texbin/pdflatex
FLAGS = -interaction=nonstopmode

all:
	$(MAKE) -C book distclean
	$(MAKE) -C book book

parts:
	$(MAKE) -C book parts

draft:
	$(MAKE) -C book draft

view:
	$(MAKE) -C book view

clean:
	$(MAKE) -C book clean

distclean:
	$(MAKE) -C book distclean

help:
	$(MAKE) -C book help

.PHONY: all parts draft view clean distclean help
