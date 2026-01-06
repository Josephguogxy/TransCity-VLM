# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    timeout_s: float = 60.0
    max_tokens: Optional[int] = None


@dataclass(frozen=True)
class SearchSettings:
    provider: str          # "serper" | "bing"
    endpoint: str
    api_key: str
    timeout_s: float = 30.0
    top_k_per_query: int = 5


@dataclass(frozen=True)
class CDSSettings:
    url: str               # e.g. "https://cds.climate.copernicus.eu/api"
    key: str               # "UID:APIKEY" or APIKEY (depends on your cdsapi)
    dataset: str = "reanalysis-era5-single-levels"
    cache_dir: str = ".cache/cds"
    request_timeout_s: int = 300


@dataclass(frozen=True)
class MySQLSettings:
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "traffic_gla"
    host: str = "localhost"
    pool_size: int = 4
    connect_timeout_s: float = 10.0


@dataclass(frozen=True)
class Settings:
    planner_llm: LLMSettings
    extractor_llm: LLMSettings
    verifier_llm: LLMSettings
    web_search: SearchSettings
    cds: CDSSettings
    mysql: MySQLSettings = MySQLSettings()

    gla_adj_path: str = str(BASE_DIR / "dataset" / "gla_rn_adj.npy")

    overpass_url: str = "https://overpass-api.de/api/interpreter"
    nominatim_url: str = "https://nominatim.openstreetmap.org/reverse"
    http_timeout_s: float = 30.0

    poi_radius_km: float = 1.0
    demo_radius_km: float = 5.0
    news_radius_km: float = 10.0
    accident_radius_km: float = 10.0


# Placeholders only; clients will error if keys are missing.
DEFAULT_SETTINGS = Settings(
    planner_llm=LLMSettings(base_url="", api_key="", model=""),
    extractor_llm=LLMSettings(base_url="", api_key="", model=""),
    verifier_llm=LLMSettings(base_url="", api_key="", model=""),
    web_search=SearchSettings(
        provider="serper",
        endpoint="https://google.serper.dev/search",
        api_key="",
        timeout_s=30.0,
        top_k_per_query=5,
    ),
    cds=CDSSettings(
        url="https://cds.climate.copernicus.eu/api",
        key="",
        dataset="reanalysis-era5-single-levels",
        cache_dir=".cache/cds",
        request_timeout_s=300,
    ),
    mysql=MySQLSettings(
        port=3306,
        user="root",
        password="",
        database="traffic_gla",
        host="localhost",
        pool_size=8,
        connect_timeout_s=10.0,
    ),
)


def get_settings() -> Settings:
    """
    Create agentic_system/local_settings.py with SETTINGS = Settings(...)
    """
    try:
        from .local_settings import SETTINGS as LOCAL_SETTINGS  # type: ignore
        return LOCAL_SETTINGS
    except Exception:
        return DEFAULT_SETTINGS


settings = get_settings()
