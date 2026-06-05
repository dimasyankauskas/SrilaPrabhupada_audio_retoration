# audio_restore — compare-first tape restoration harness
#
# Targets:
#   make setup    one-time: create .venv, install deps
#   make inspect  read source, write report + spectrogram
#   make compare  run every candidate, score, rank
#   make multi    run every candidate on every source in samples/source/,
#                 write a side-by-side HTML report to reports/multi/comparison.html
#   make best     show the current best candidate from reports/compare.md
#   make clean    remove stages/ and reports/ (preserves samples/source/)

PY ?= python3
VENV ?= .venv
ACT := . $(VENV)/bin/activate &&

.PHONY: setup inspect compare multi best clean help

help:
	@echo "make setup    — create venv and install requirements"
	@echo "make inspect  — characterize the source (stages/00--inspect/)"
	@echo "make compare  — run every candidate and write reports/compare.md"
	@echo "make multi    — run every candidate on every source, side-by-side HTML"
	@echo "make best     — print the current top-ranked candidate"
	@echo "make clean    — remove stages/ and reports/"

setup:
	@command -v uv >/dev/null 2>&1 || { echo "uv not found. Install: brew install uv"; exit 1; }
	uv venv $(VENV) --python 3.13
	uv pip install --python $(VENV)/bin/python -r requirements.txt
	@echo "venv ready. Activate with: source $(VENV)/bin/activate"

inspect:
	$(ACT) $(PY) tools/00_inspect.py

compare:
	$(ACT) $(PY) tools/99_compare.py

multi:
	$(ACT) $(PY) tools/run_multi.py

best:
	@test -f reports/compare.md || { echo "no compare.md yet — run 'make compare'"; exit 1; }
	@head -n 25 reports/compare.md

clean:
	rm -rf stages reports
	@echo "removed stages/ and reports/"
