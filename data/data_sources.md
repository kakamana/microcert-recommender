# Data Sources — H7 Micro-Certification Recommender

## Primary (synthetic)
| # | Source | Notes |
|---|---|---|
| 1 | `src/microcert_rec/data.py` | Deterministic, seed=42 |

## Real-world drop-ins
| Source | URL | Use |
|---|---|---|
| Coursera Course Catalogue | https://www.coursera.org/ | Real cert metadata |
| LinkedIn Learning Catalogue | https://learning.linkedin.com/ | Real cert metadata |
| edX Course Catalogue | https://www.edx.org/ | Real cert metadata |
| Internal LMS export | (private) | (learner_id, cert_id, event_type, rating) |

## Schema for drop-in
- `learners.parquet`: `learner_id, skill_<n>` columns (40 binary)
- `certs.parquet`: `cert_id, title, issuer, skills_taught (text), hours, cost`
- `interactions.parquet`: `learner_id, cert_id, event_type {enroll, complete, rated}, rating, ts`

## Attribution
Public catalogues retain their respective licences; this repo ships only synthetic data.
