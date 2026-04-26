"""FastAPI for the Micro-Certification Recommender.

Endpoints:
    GET  /health
    POST /recommend  - top-K certs for a learner skill set (+ optional learner_id)
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Micro-Certification Recommender", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    learner_id: str | None = Field(default=None, description="Optional ID for warm-start CF tower")
    learner_skills: list[str] = Field(..., min_length=1)
    k: int = Field(default=10, ge=1, le=50)
    beta: float = Field(default=0.6, ge=0.0, le=1.0,
                        description="Blend weight: collaborative vs content tower")


class CertResult(BaseModel):
    cert_id: str
    title: str
    issuer: str
    hours: float
    cost: float
    relevance: float
    cf_score: float
    content_score: float
    reason: str
    rank: int


class RecommendResponse(BaseModel):
    learner_id: str | None
    used_collaborative_tower: bool
    items: list[CertResult]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    try:
        from microcert_rec.serve import recommend as _serve_recommend
        items = _serve_recommend(
            learner_skills=req.learner_skills,
            learner_id=req.learner_id,
            k=req.k,
            beta=req.beta,
        )
        used_cf = req.learner_id is not None
    except FileNotFoundError:
        # Pre-training stub so the wiring is testable without artefacts.
        items = [
            dict(
                cert_id=f"C-STUB-{i:03d}",
                title=f"Stub Cert #{i}",
                issuer="Coursera",
                hours=10.0 + i,
                cost=49.0 + 5 * i,
                relevance=round(0.85 - 0.04 * i, 4),
                cf_score=0.0,
                content_score=round(0.85 - 0.04 * i, 4),
                reason=f"matches your {', '.join(req.learner_skills[:2])}",
                rank=i + 1,
            )
            for i in range(min(req.k, 10))
        ]
        used_cf = False
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    return RecommendResponse(
        learner_id=req.learner_id,
        used_collaborative_tower=used_cf,
        items=[CertResult(**it) for it in items],
    )
