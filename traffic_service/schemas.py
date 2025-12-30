# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# -------------------------
# Input / request
# -------------------------
class UserRequest(BaseModel):
    # Allow requests without lat/lon/time.
    lat: Optional[float] = None
    lon: Optional[float] = None
    local_start: Optional[str] = None  # "YYYY-MM-DD HH:MM"
    local_end: Optional[str] = None

    question: str

    # Injected by service/CLI: request time in UTC.
    request_time_utc: Optional[str] = None  # ISO8601 e.g. "2025-12-16T12:34:56Z"

    sensor_id: Optional[str] = None
    lanes: Optional[int] = None
    direction: Optional[str] = None
    freeway: Optional[str] = None

    traffic_flow: Optional[list[float]] = None
    history_average_flow: Optional[list[float]] = None


# -------------------------
# Orchestrator plan schema
# -------------------------
class PlanStep(BaseModel):
    step_id: str = Field(..., description="Unique step id, e.g., 'weather'.")
    agent: str = Field(..., description="Agent name, e.g., 'WeatherAgent'.")
    action: str = Field(..., description="Action/method name in the agent.")
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    output_key: str = Field(..., description="Where to store output in context.")


class Plan(BaseModel):
    version: str = "v1"
    steps: list[PlanStep]
    final_outputs: list[str] = Field(default_factory=list)


# -------------------------
# Agent outputs / normalized chunks
# -------------------------
class PlaceInfo(BaseModel):
    display_name: str
    city: Optional[str] = None
    county: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None


class WeatherHourly(BaseModel):
    # Hourly aligned to *local time* in output
    start_local: str
    end_local: str
    temperature_c: list[float] = Field(default_factory=list)
    wind_speed_ms: list[float] = Field(default_factory=list)
    wind_direction_deg: list[float] = Field(default_factory=list)


class POISummary(BaseModel):
    radius_km: float
    counts_by_category: dict[str, int] = Field(default_factory=dict)


class DemographicsSummary(BaseModel):
    radius_km: float
    population_density: Optional[float] = None   # people / km^2 (suggested)
    housing_density: Optional[float] = None      # units / km^2 (suggested)
    notes: Optional[str] = None


class SearchResult(BaseModel):
    title: str
    snippet: str
    url: str
    source: Optional[str] = None
    published: Optional[str] = None  # ISO date-time if available


class NewsEvent(BaseModel):
    event_time: Optional[str] = None  # ISO
    location_text: Optional[str] = None
    summary: str
    category: Optional[str] = None
    tone: Optional[float] = None  # optional sentiment-like score
    source_url: Optional[str] = None
    confidence: float = 0.5


class AccidentEvent(BaseModel):
    event_time: Optional[str] = None
    location_text: Optional[str] = None
    collision_type: Optional[str] = None
    fatalities: Optional[int] = None
    injuries: Optional[int] = None
    conditions: Optional[str] = None
    summary: str
    source_url: Optional[str] = None
    confidence: float = 0.5


class Chunks(BaseModel):
    POI: list[str] = Field(default_factory=list)
    News: list[str] = Field(default_factory=list)
    Accident: list[str] = Field(default_factory=list)
    HopSensor: list[str] = Field(default_factory=list)
    HopBA: list[str] = Field(default_factory=list)


# -------------------------
# Dataset-style output
# -------------------------
class ChatContent(BaseModel):
    # Supports multimodal content.
    type: Literal["text", "image"] = "text"
    text: Optional[str] = None
    image: Optional[str] = None
    mime_type: Optional[str] = None  # e.g. "image/png"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: list[ChatContent]


class DatasetRecord(BaseModel):
    task: str = "reason"
    messages: list[ChatMessage]
    chunks: Chunks
    thinking: Optional[list[str]] = None
