PAPER = paper
LATEX = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX = bibtex
PYTHON = python3

.PHONY: all paper figures tests clean

all: tests figures paper

paper: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

figures: tests
	$(PYTHON) code/04_master_figure.py

tests:
	$(PYTHON) code/01_radial_profile.py
	$(PYTHON) code/02_time_dilation.py
	$(PYTHON) code/03_photon_deflection.py

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).bbl $(PAPER).blg $(PAPER).out $(PAPER).toc
