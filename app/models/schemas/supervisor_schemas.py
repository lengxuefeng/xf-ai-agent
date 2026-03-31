# -*- coding: utf-8 -*-
"""Supervisor planning and reflection schemas."""
from __future__ import annotations

from typing import List

from pydantic import Field

from models.schemas.base import BaseSchema
from config.constants.workflow_constants import RouteStrategy


class IntentDecision(BaseSchema):
    intent: str = "CHAT"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    is_complex: bool = False
    direct_answer: str = ""


class DomainDecision(BaseSchema):
    data_domain: str = "GENERAL"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = "llm"


class PlannerTaskDecision(BaseSchema):
    id: str = "t1"
    agent: str = "CHAT"
    input: str = ""
    depends_on: List[str] = Field(default_factory=list)


class PlannerDecision(BaseSchema):
    tasks: List[PlannerTaskDecision] = Field(default_factory=list)


class ReflectionDecision(BaseSchema):
    continue_execution: bool = False
    summary: str = ""
    tasks: List[PlannerTaskDecision] = Field(default_factory=list)


class RequestAnalysisDecision(BaseSchema):
    candidate_agents: List[str] = Field(default_factory=list)
    candidate_domains: List[str] = Field(default_factory=list)
    is_multi_intent: bool = False
    is_multi_domain: bool = False
    has_dependency_hint: bool = False
    route_strategy: str = RouteStrategy.SINGLE_DOMAIN.value
    reason: str = "single_domain"
