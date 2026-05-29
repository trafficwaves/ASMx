#!/usr/bin/env bash
# Build data/README.pdf from data/README.tex.
# Runs pdflatex twice so the table of contents and cross-references resolve,
# then removes the auxiliary files it produces.

set -euo pipefail

cd "$(dirname "$0")"

TEX_FILE="README.tex"
PDF_FILE="README.pdf"

if ! command -v pdflatex >/dev/null 2>&1; then
    echo "Error: pdflatex not found on PATH. Install TeX Live (e.g. 'sudo apt install texlive-latex-extra texlive-fonts-extra')." >&2
    exit 1
fi

if [[ ! -f "$TEX_FILE" ]]; then
    echo "Error: $TEX_FILE not found in $(pwd)." >&2
    exit 1
fi

echo "[1/2] First pdflatex pass..."
pdflatex -interaction=nonstopmode -halt-on-error "$TEX_FILE" >/dev/null

echo "[2/2] Second pdflatex pass (resolves ToC and cross-references)..."
pdflatex -interaction=nonstopmode -halt-on-error "$TEX_FILE" >/dev/null

echo "Cleaning auxiliary files..."
rm -f README.aux README.log README.out README.toc

if [[ -f "$PDF_FILE" ]]; then
    echo "Done. Output: $(pwd)/$PDF_FILE"
else
    echo "Error: $PDF_FILE was not produced. Re-run with 'pdflatex $TEX_FILE' to see the error." >&2
    exit 1
fi
