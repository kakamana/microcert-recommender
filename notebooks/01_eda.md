# Notebook 01 — EDA

>>> `from microcert_rec.data import load_all; learners, certs, inter = load_all()`

## 1. Sparsity
- |Learners|, |Certs|, |Interactions|, density.

## 2. Skewness
- Top-20 most-enrolled certs (popularity head).
- Long-tail histogram of cert enrolment counts.

## 3. Learner skill density
- # skills per learner; histogram.
- Skills-per-cert histogram.

## 4. Hypotheses
1. Popularity baseline will be hard to beat on top-1.
2. Content tower will dominate cold-start.
3. Combined score wins on Recall@10 + coverage.
