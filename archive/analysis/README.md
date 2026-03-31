# Archived Analysis Modules

This directory contains analysis modules that are kept for historical reference
but are intentionally **not** part of the importable `ffbayes` package.

## Why this exists

Some experiments were checked into `src/ffbayes/analysis/` during iteration and
were later superseded or left in a known-broken state. Keeping them under
`archive/` prevents accidental imports while preserving the work for future
porting or review.

## Contents

- `BROKEN_bayesian_hierarchical_ff_unified.py`: legacy experiment retained as-is.

