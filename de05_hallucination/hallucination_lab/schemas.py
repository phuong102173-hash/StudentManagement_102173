from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .text import tokens

Nonempty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Method(StrEnum):
    BASELINE = "evidence_baseline"
    FACTSCORE = "factscore"
    SELFCHECK = "selfcheckgpt"
    HHEM = "hhem"
    JUDGE = "llm_judge"
    NLI = "nli"
    ENSEMBLE = "ensemble"


class DetectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: Nonempty = Field(max_length=6000)
    context: str = Field(default="", max_length=40000)
    question: str = Field(default="", max_length=4000)
    samples: list[Annotated[Nonempty, Field(max_length=12000)]] = Field(default_factory=list, max_length=10)
    method: Method = Method.BASELINE
    threshold: Score = 0.5

    @model_validator(mode="after")
    def meaningful_text(self):
        if not tokens(self.claim):
            raise ValueError("claim must contain words or numbers")
        return self


class Evidence(BaseModel):
    quote: str
    source: str
    relation: Literal["support", "contradiction", "related", "sample"]


class Finding(BaseModel):
    claim: str
    score: Score | None
    verdict: Literal["supported", "unsupported", "contradicted", "unknown"]
    explanation: str
    error_type: Literal["factual_error", "unfaithful_summary", "fabricated_citation", "unsupported", "none"]
    evidence: list[Evidence] = Field(default_factory=list)


class DetectionResult(BaseModel):
    method: Method
    is_hallucination: bool | None
    score: Score | None
    confidence: Score | None
    confidence_kind: str = "uncalibrated_score"
    threshold: Score
    explanation: str
    evidence: list[Evidence] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    components: list["DetectionResult"] = Field(default_factory=list)
    latency_ms: float = 0


class JudgeOutput(BaseModel):
    """Validate every external model answer before it affects a verdict."""

    model_config = ConfigDict(extra="forbid")
    verdict: Literal["supported", "unsupported", "contradicted", "unknown"]
    hallucination_score: Score | None
    explanation: Nonempty = Field(max_length=4000)
    error_type: Literal["factual_error", "unfaithful_summary", "fabricated_citation", "unsupported", "none"]
    evidence_quotes: list[str] = Field(max_length=8)

    @model_validator(mode="after")
    def consistent(self):
        if self.verdict == "unknown" and self.hallucination_score is not None:
            raise ValueError("unknown verdict must have a null score")
        if self.verdict != "unknown" and self.hallucination_score is None:
            raise ValueError("a verdict requires a score")
        if self.verdict == "supported" and self.hallucination_score >= 0.5:
            raise ValueError("supported verdict conflicts with the score")
        if self.verdict in {"unsupported", "contradicted"} and self.hallucination_score < 0.5:
            raise ValueError("hallucination verdict conflicts with the score")
        return self
