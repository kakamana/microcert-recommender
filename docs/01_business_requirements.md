# Business Requirements — Micro-Certification Recommender

## 1. Problem Statement
Internal L&D libraries hold thousands of micro-certifications (Coursera, LinkedIn Learning, in-house cohorts, vendor badges). Most learners pick from a generic top-10. Personalisation today is rule-based ("if Sales -> Sales course") and doesn't account for the learner's existing skill graph or peer behaviour. Outcome: low completion, low transfer, low ROI per L&D dollar.

## 2. Stakeholders
| Role | Interest | Success criterion |
|---|---|---|
| Learner / Employee | Recommendations that fit my actual skills | Top-K relevance >= 60% (self-reported) |
| L&D Catalogue Owner | Surface long-tail certs | Catalogue coverage > 50% |
| HRBP | Map team upskill paths | Recall@10 on holdout >= 0.35 |
| Finance | Don't pay for unused seats | Per-cert enrolment-to-completion ratio improves |

## 3. Business Objectives
1. **Recall@10 >= 0.35** on a held-out interaction split.
2. **Cold-start MRR >= 0.20** for learners with no prior interactions (content tower carries it).
3. **Catalogue coverage >= 50%** of certs reachable in the top-K of *some* learner.
4. Each recommendation returns a short **reason** ("matches your SQL + Pandas skills" / "popular among peers in Data roles").

## 4. KPIs
| KPI | Definition | Target |
|---|---|---|
| Recall@10 | True positive enrolments in top-10 | >= 0.35 |
| nDCG@10 | Rank-discounted relevance | >= 0.40 |
| Coverage | Unique certs reachable in top-K across learners | >= 50% |
| Cold-start MRR | MRR for learners with 0 history | >= 0.20 |

## 5. Scope
**In scope:** synthetic learners (40 skills each), 500 cert metadata, 25k learner-cert interaction events.
**Out of scope:** curriculum sequencing (multi-cert paths); live LMS streaming events.

## 6. Constraints
- CPU-only; entire model train + dump < 60 seconds on 2k×500.
- Stateless inference: load SVD + TF-IDF artefacts once, < 50 ms per request.

## 7. Risks
| Risk | Mitigation |
|---|---|
| Popularity bias | Re-rank to enforce coverage budget |
| Cold-start (new learners or new certs) | Content tower fallback |
| Overfitting to ratings noise | TruncatedSVD with k tuned on validation |
