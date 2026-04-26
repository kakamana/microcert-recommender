# Evaluation Plan — Micro-Certification Recommender

## 1. Held-out split
Per-learner **leave-last-N-out** (N=2) on the timestamp-sorted interaction stream. Train on the rest.

## 2. Primary scorecard
| Model | Recall@10 | nDCG@10 | Coverage | Cold-start MRR |
|---|---|---|---|---|
| Popularity baseline | – | – | – | – |
| Collaborative tower only (SVD k=32) | – | – | – | – |
| Content tower only (TF-IDF) | – | – | – | – |
| **Combined β=0.6** | – | – | – | – |

## 3. Cold-start
Mask all interactions for a random 10% of learners. Score using only the content tower. Report MRR + Hit@10.

## 4. Coverage / fairness to long-tail certs
- % of catalogue appearing in any learner's top-K.
- Gini coefficient over per-cert recommendation counts (lower = fairer to the long tail).

## 5. Slice analysis
- By learner skill-density bucket (low/med/high # of known skills)
- By cert hours bucket (short / medium / long)
- By issuer

## 6. Robustness
- Drop 50% of interactions → measure Recall@10 decay.
- Add 100 synthetic noisy interactions per learner → measure rank stability.

## 7. Business impact
- Estimate uplift in cert completion rate vs popularity-only.
- Translate to "freed budget" assuming $X/cert seat.

## 8. Deployment readiness checklist
- [ ] Recall@10 >= 0.35 on holdout
- [ ] Cold-start MRR >= 0.20
- [ ] Coverage >= 50%
- [ ] /recommend returns reason chip
- [ ] Re-rank for coverage active by default
- [ ] Model card published
