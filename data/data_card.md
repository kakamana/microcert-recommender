# Data Card — H7 Micro-Certification Recommender

## Dataset composition

| Layer | Source | Shape | Purpose |
|---|---|---|---|
| Synthetic learners | `src/microcert_rec/data.py` | 2,000 × 40 | Learner skill vectors |
| Synthetic certs | `src/microcert_rec/data.py` | 500 × {issuer, skills_taught, hours, cost} | Cert catalogue |
| Synthetic interactions | `src/microcert_rec/data.py` | 25,000 events | enroll / complete / rated |

## Files
- `data/processed/learners.parquet`
- `data/processed/certs.parquet`
- `data/processed/interactions.parquet`

## Skill vocabulary (40)
Drawn from the same controlled list used in H6 (data + cloud + soft-skills overlap), persisted as a tuple in `data.py::SKILL_VOCAB`.

## Known biases
- The synthetic generator induces a clear "popularity head" so the recommender has something to fight (popularity baseline is non-trivial).
- Issuer distribution is intentionally skewed (a few big issuers + a long tail).

## PII
None. Learner IDs are surrogate keys.

## Reproducing
```bash
python -m microcert_rec.data
```
Deterministic seed = 42.
