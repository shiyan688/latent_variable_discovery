#!/usr/bin/env bash
set -euo pipefail

project_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
paper_root="$project_root/paper"
cache_root="$project_root/runs/_runtime_cache"

mkdir -p \
  "$cache_root/texmf-var" \
  "$cache_root/texmf-config" \
  "$cache_root/texfonts"

export TEXINPUTS="$paper_root/iclr2027:${TEXINPUTS:-}"
export BSTINPUTS="$paper_root/iclr2027:${BSTINPUTS:-}"
export TEXMFVAR="$cache_root/texmf-var"
export TEXMFCONFIG="$cache_root/texmf-config"
export VARTEXFONTS="$cache_root/texfonts"
export SOURCE_DATE_EPOCH=1787987400
export FORCE_SOURCE_DATE=1

cd "$paper_root"
pdflatex -interaction=nonstopmode -halt-on-error iclr2027_draft.tex
bibtex iclr2027_draft
pdflatex -interaction=nonstopmode -halt-on-error iclr2027_draft.tex
pdflatex -interaction=nonstopmode -halt-on-error iclr2027_draft.tex
pdflatex -interaction=nonstopmode -halt-on-error iclr2027_draft.tex
