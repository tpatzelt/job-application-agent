from __future__ import annotations

from pydantic import BaseModel, Field


class SearchQueries(BaseModel):
    queries: list[str] = Field(default_factory=list)


class JobEvaluation(BaseModel):
    score: int
    reason: str


class SearchPlan(BaseModel):
    target_roles: list[str] = Field(default_factory=list)
    key_skills: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    strategy: str = ""


class Reflection(BaseModel):
    assessment: str = ""
    effective_queries: list[str] = Field(default_factory=list)
    ineffective_queries: list[str] = Field(default_factory=list)
    adjustments: list[str] = Field(default_factory=list)


class JobResult(BaseModel):
    title: str
    company: str
    url: str
    score: int
    reason: str
    status: str
