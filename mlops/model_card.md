# Model Card — Micro-Certification Recommender

## Intended use
Personalised top-K micro-cert recommendations for L&D learners. Each result carries a one-line reason and ROI hint (hours, cost). Advisory; learners pick freely.

## Training data
Synthetic learners (2,000 × 40 binary skills), certs (500 with `skills_taught` text), interactions (~25,000 enrol / complete / rated events). Drop-in: any LMS export with the same shape.

## Model family
- Collaborative tower: TruncatedSVD, k=32
- Content tower: TF-IDF over `skills_taught`, ngrams (1, 2)
- Combined: weighted dot product, β=0.6 default

## Metrics (target)
| Metric | Target |
|---|---|
| Recall@10 | >= 0.35 |
| nDCG@10 | >= 0.40 |
| Coverage | >= 50% |
| Cold-start MRR | >= 0.20 |

## Limitations
- TruncatedSVD is symmetric in the way it treats the implicit signal; ratings vs enrolments share scoring weight via the EVENT_WEIGHT lookup.
- TF-IDF on `skills_taught` is bag-of-words; sentence-transformer embeddings would help with synonymy.
- Popularity bias is partially controlled by the coverage re-rank.

## Ethical considerations
- No protected attributes in either tower.
- Recommendations advisory, not mandatory.

## Retraining
- Weekly. Watch nDCG@10 + coverage; trigger refit if either drops > 5 pts.
