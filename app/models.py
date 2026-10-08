from pydantic import BaseModel, Field
from typing import Literal


class ResearchQuestion(BaseModel):
    question: str
    purpose: str


class ResearchPlan(BaseModel):
    original_query: str
    questions: list[ResearchQuestion]


class Source(BaseModel):
    title: str
    url: str
    content: str = ""


class Evidence(BaseModel):
    research_question: str
    claim: str
    evidence: str
    source_url: str
    source_title: str
    confidence: float = Field(ge=0, le=1)
    evidence_type: Literal[
        "fact",
        "inference",
        "opinion"
    ]


class ResearchReport(BaseModel):
    question: str
    executive_summary: str
    key_findings: list[str]
    evidence: list[Evidence]
    counterarguments: list[str]
    conclusion: str

class Source(BaseModel):
    title: str
    url: str
    content: str = ""
    source_quality: str = "unknown"