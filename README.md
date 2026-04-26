# Micro-Certification Recommender

> **Two-tower-style recommender mapping learner skill vectors to micro-certifications with relevance + ROI hints.** A collaborative tower (TruncatedSVD over the learner-cert interaction matrix) and a content tower (TF-IDF over `skills_taught`) combined via weighted dot product.

![Python](https://img.shields.io/badge/python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688) ![Next.js](https://img.shields.io/badge/Next.js-14-black) ![License](https://img.shields.io/badge/license-MIT-green)

## Why this project
- L&D libraries balloon to thousands of micro-certs. Most are never enrolled; a handful are over-enrolled. Learners get either *too much choice* or *generic top-10s*.
- This project gives a learner-personalised top-K with two complementary signals: what people-like-you finished (collaborative) and what overlaps with your skills (content). The output also surfaces a small ROI-hint payload (hours, cost, expected uplift band).

## Table of contents
- [Business Requirements](./docs/01_business_requirements.md)
- [Feasibility Study](./docs/02_feasibility_study.md)
- [Methodology — Two-tower + TruncatedSVD](./docs/03_methodology.md)
- [Evaluation Plan](./docs/04_evaluation.md)
- [Data card](./data/data_card.md) - [Data sources](./data/data_sources.md)
- [Notebooks](./notebooks/) - [Source](./src/microcert_rec/) - [API](./api/main.py) - [UI](./ui/app/page.tsx)

## Headline results (target)

| Metric | Popularity baseline | Two-tower | Target |
|---|---|---|---|
| Recall@10 | 0.18 | **0.42** | > 0.35 |
| nDCG@10 | 0.20 | **0.46** | > 0.40 |
| Coverage (% catalog reachable) | 8% | **64%** | > 50% |
| Cold-start MRR (new learner) | 0.05 | **0.28** | > 0.20 |

## Quickstart

```bash
pip install -e ".[dev]"
python -m microcert_rec.data         # generate learners, certs, interactions parquet
python -m microcert_rec.models       # fit TruncatedSVD + TF-IDF towers, save artifacts
uvicorn api.main:app --reload
cd ui && npm install && npm run dev
```

## Stack
Python - pandas - scikit-learn (TruncatedSVD + TF-IDF) - FastAPI - Next.js - Tailwind

## Author
Asad - MADS @ University of Michigan - Dubai HR
