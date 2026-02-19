from __future__ import annotations

from pydantic import BaseModel, Field


class SearchQueries(BaseModel):
    queries: list[str] = Field(default_factory=list)


class JobEvaluation(BaseModel):
    score: int
    reason: str


class JobResult(BaseModel):
    title: str
    company: str
    url: str
    score: int
    reason: str
    status: str
