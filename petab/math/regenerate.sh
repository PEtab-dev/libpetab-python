#!/usr/bin/env sh
# This script regenerates the ANTLR parser and lexer for PEtab math expressions
set -eou > /dev/null

# ANTLR version
# IMPORTANT: when updating this, also update the version for
# `antlr4-python3-runtime` in `setup.py`
antlr_version="4.13.1"

pip show antlr4-tools > /dev/null || pip3 install antlr4-tools

cd "$(dirname "$0")"

antlr4 -v $antlr_version \
  -Dlanguage=Python3 \
  -visitor \
  -no-listener \
  -o _generated \
  PetabMathExprParser.g4 PetabMathExprLexer.g4

echo "# auto-generated" > _generated/__init__.py
