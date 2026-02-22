#!/bin/bash
# Moves LaTeX auxiliary files to the latex_files/ subfolder after a build.
cd "$(dirname "$0")"
mkdir -p latex_files
for ext in aux bbl blg fls fdb_latexmk log toc lot lof out acn acr alg glg glo gls idx ind ilg nav snm vrb bcf run.xml synctex.gz xmpi; do
  for f in *.$ext; do
    [ -f "$f" ] && mv "$f" latex_files/
  done
done
