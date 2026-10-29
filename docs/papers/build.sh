#!/usr/bin/env bash
# Builds the paper's PDFs: docs/papers/build.sh [elsevier] [ieee] [highlights] [final]   (default: the first three)
# final: the submission build of both versions (main_final.pdf), in which every part whose result has not come back
# is left out and any unfilled slot is an error.
# Needs pdflatex and bibtex with the elsarticle and IEEEtran classes (both on Overleaf and in TeX Live / MiKTeX).
# Run docs/papers/fill.py first, so that the result slots are filled.
set -e
cd "$(dirname "$0")"
targets=${*:-elsevier ieee highlights}
for t in $targets; do
  case $t in
    elsevier|ieee)
      (cd "$t" && pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log \
         && bibtex main >> build.log && pdflatex -interaction=nonstopmode main.tex >> build.log \
         && pdflatex -interaction=nonstopmode main.tex >> build.log) \
        || { echo "$t: build failed, see docs/papers/$t/build.log"; exit 1; }
      echo "$t: docs/papers/$t/main.pdf ($(grep -c 'Placeholder\|PLACEHOLDER' "$t/main.log" 2>/dev/null || true) placeholder warnings in the log)";;
    final)
      for v in elsevier ieee; do
        (cd "$v" && J=main_final && pdflatex -interaction=nonstopmode -halt-on-error -jobname=$J "\\def\\finalbuild{}\\input{main.tex}" > $J.build.log \
           && bibtex $J >> $J.build.log && pdflatex -interaction=nonstopmode -jobname=$J "\\def\\finalbuild{}\\input{main.tex}" >> $J.build.log \
           && pdflatex -interaction=nonstopmode -halt-on-error -jobname=$J "\\def\\finalbuild{}\\input{main.tex}" >> $J.build.log) \
          || { echo "final $v: build failed, see docs/papers/$v/main_final.build.log"; exit 1; }
        echo "final $v: docs/papers/$v/main_final.pdf"
      done;;
    highlights)
      (cd elsevier && pdflatex -interaction=nonstopmode -halt-on-error highlights.tex > highlights.log) \
        && echo "highlights: docs/papers/elsevier/highlights.pdf";;
  esac
done
