from __future__ import annotations

from pydantic import BaseModel, Field


class SearchQueries(BaseModel):
    queries: list[str] = Field(default_factory=list)


class JobEvaluation(BaseModel):
    score: int
    reason: str
    # None when the model didn't say; False triggers rejection regardless
    # of score (preferred-location mismatch).
    location_match: bool | None = None


class SearchPlan(BaseModel):
    target_roles: list[str] = Field(default_factory=list)
    key_skills: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    target_companies: list[str] = Field(default_factory=list)
    strategy: str = ""


class Reflection(BaseModel):
    assessment: str = ""
    effective_queries: list[str] = Field(default_factory=list)
    ineffective_queries: list[str] = Field(default_factory=list)
    adjustments: list[str] = Field(default_factory=list)


class IntakeExtraction(BaseModel):
    """Search parameters distilled from a user's uploaded documents,
    plus follow-up questions for information the documents don't cover."""

    job_titles: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    # Language the user wants job postings in; empty unless they stated one.
    language: str = ""
    questions: list[str] = Field(default_factory=list)


class JobResult(BaseModel):
    title: str
    company: str
    url: str
    score: int
    reason: str
    status: str
