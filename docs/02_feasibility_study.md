# Feasibility Study — Micro-Certification Recommender

## 1. Data feasibility
- **Synthetic generator:** 2,000 learner skill vectors over 40 skills; 500 cert metadata records (issuer, hours, cost, `skills_taught`); 25,000 interaction events (`enrolled / completed / rated`).
- **Real-world drop-in:** any LMS export with `(learner_id, cert_id, event_type, rating)` plus a cert metadata table fits the same schema.

## 2. Technical feasibility
- **Algorithmic shortlist**
  - Collaborative tower: TruncatedSVD on the implicit (learner, cert) matrix (main); ALS/implicit (stretch).
  - Content tower: TF-IDF over `skills_taught` text (main); sentence-transformer embedding (stretch, GPU-bound).
  - Combiner: weighted sum of cosine scores; weight tuned on validation.
- **Compute:** 1 CPU; full pipeline < 1 minute on 2k×500.
- **Serving:** pre-computed learner factors + cert TF-IDF index; online lookup is dot product.

## 3. Economic feasibility
| Line item | Monthly cost |
|---|---|
| 1× small container | ~$8 |
| Storage | ~$1 |
| **Total** | **~$9 / mo** |

**Value:** lifting cert completion by 10 percentage points on a 5,000-employee L&D budget materially shifts ROI per cert hour.

## 4. Operational feasibility
- Refit weekly. Online inference reads two joblib artefacts.
- Re-rank step enforces coverage budget post-scoring.

## 5. Ethical / legal feasibility
- No PII; learner IDs are surrogate keys.
- Recommendations are advisory; learners pick freely.
- No protected-attribute features in the towers.

## 6. Recommendation
**Go.** Cheap, deterministic, dependency-light. Two-tower keeps the cold-start path open without sacrificing personalisation.
