# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from traffic_service.core.context import ExecutionContext
from traffic_service.core.executor import Executor
from traffic_service.core.plan_builder import PlanBuilder, FeatureFlags
from traffic_service.schemas import UserRequest


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class RunOptions:
    with_flow: bool = True
    with_satellite: bool = True
    image_store: str = "file"   # base64/file/none
    print_prompt: bool = False


class QuestionRunner:
    def __init__(self) -> None:
        pass

    def build_agents(self, *, image_store: str = "file") -> Dict[str, Any]:
        # Imports follow the new package layout.
        from traffic_service.agents.bootstrap.nl_parse import NLRequestParserAgent
        from traffic_service.agents.bootstrap.geocode_forward import ForwardGeocodeAgent
        from traffic_service.agents.bootstrap.time_window import TimeWindowResolverAgent
        from traffic_service.agents.bootstrap.normalize import RequestNormalizerAgent

        from traffic_service.agents.traffic.nearest_sensor import NearestSensorAgent
        from traffic_service.agents.traffic.traffic_flow import TrafficFlowAgent

        from traffic_service.agents.geo.geocode_reverse import ReverseGeocodeAgent
        from traffic_service.agents.osm.roads import RoadContextAgent
        from traffic_service.agents.osm.poi import POIAgent

        from traffic_service.agents.weather.weather import WeatherAgent
        from traffic_service.agents.demographics.demographics import DemographicsAgent

        from traffic_service.agents.web.query_generator import QueryGeneratorAgent
        from traffic_service.agents.web.web_search_agent import WebSearchAgent
        from traffic_service.agents.web.events import NewsExtractorAgent, AccidentExtractorAgent, VerifiedEventConverterAgent

        from traffic_service.agents.satellite.fetch_gibs import SatelliteImageAgent
        from traffic_service.agents.satellite.store_mysql import SatelliteImageStoreMySQLAgent

        from traffic_service.agents.record.record_builder import RecordBuilder

        return {
            "NLRequestParserAgent": NLRequestParserAgent(),
            "ForwardGeocodeAgent": ForwardGeocodeAgent(),
            "TimeWindowResolverAgent": TimeWindowResolverAgent(),
            "RequestNormalizerAgent": RequestNormalizerAgent(),

            "NearestSensorAgent": NearestSensorAgent(),
            "TrafficFlowAgent": TrafficFlowAgent(),

            "ReverseGeocodeAgent": ReverseGeocodeAgent(),
            "RoadContextAgent": RoadContextAgent(),
            "POIAgent": POIAgent(),

            "WeatherAgent": WeatherAgent(),
            "DemographicsAgent": DemographicsAgent(),

            "QueryGeneratorAgent": QueryGeneratorAgent(),
            "WebSearchAgent": WebSearchAgent(),
            "NewsExtractorAgent": NewsExtractorAgent(fetch_fulltext=False),
            "AccidentExtractorAgent": AccidentExtractorAgent(fetch_fulltext=False),
            "VerifiedEventConverterAgent": VerifiedEventConverterAgent(),

            "SatelliteImageAgent": SatelliteImageAgent(store=image_store),
            "SatelliteImageStoreMySQLAgent": SatelliteImageStoreMySQLAgent(),

            "RecordBuilder": RecordBuilder(),
        }

    def close_agents(self, agents: Dict[str, Any]) -> None:
        for a in agents.values():
            try:
                if hasattr(a, "close") and callable(a.close):
                    a.close()
            except Exception:
                pass

    def print_prompt_from_dataset_record(self, payload: dict) -> None:
        msgs = payload.get("messages") or []
        for m in msgs:
            if (m.get("role") or "") == "user":
                contents = m.get("content") or []
                for c in contents:
                    if c.get("type") == "text" and c.get("text"):
                        print("\n===== PROMPT (user text) =====\n")
                        print(c["text"])
                        print("\n===== END PROMPT =====\n")
                        return
        print("[WARN] Could not find user text prompt in dataset_record.messages")

    async def run_request(self, req: UserRequest, *, opts: RunOptions) -> dict:
        agents = self.build_agents(image_store=opts.image_store)
        executor = Executor(agents=agents)

        try:
            flags = FeatureFlags(
                flow=bool(opts.with_flow),
                satellite=bool(opts.with_satellite),
                context=True,
                web_events=True,
            )
            plan = PlanBuilder().build(req, flags=flags)

            ctx = ExecutionContext(request=req)
            ctx = await executor.execute(plan, ctx)

            record = ctx.outputs.get("dataset_record")
            if record is None:
                return {"error": "dataset_record missing", "outputs_keys": list(ctx.outputs.keys())}

            payload = record.model_dump(mode="json", exclude_none=True) if hasattr(record, "model_dump") else record

            if opts.print_prompt:
                self.print_prompt_from_dataset_record(payload)

            return payload
        finally:
            self.close_agents(agents)

    async def run_question(self, question: str, *, opts: RunOptions) -> dict:
        req = UserRequest(
            question=question,
            request_time_utc=utc_now_iso(),
            lat=None,
            lon=None,
            local_start=None,
            local_end=None
        )
        return await self.run_request(req, opts=opts)
