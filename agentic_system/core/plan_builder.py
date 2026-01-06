# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from dataclasses import dataclass

from agentic_system.config import settings
from agentic_system.schemas import Plan, PlanStep, UserRequest


@dataclass(frozen=True)
class FeatureFlags:
    flow: bool = True
    satellite: bool = True
    context: bool = True
    web_events: bool = True


class PlanBuilder:
    def build(self, req: UserRequest, *, flags: FeatureFlags) -> Plan:
        steps: list[PlanStep] = []

        def add(step_id: str, agent: str, action: str, output_key: str, depends_on=None, **args):
            steps.append(
                PlanStep(
                    step_id=step_id,
                    agent=agent,
                    action=action,
                    args=args,
                    depends_on=list(depends_on or []),
                    output_key=output_key,
                )
            )

        # ---- bootstrap (required) ----
        add("nl_parse", "NLRequestParserAgent", "parse_nl", "parsed_nl")
        add("geo", "ForwardGeocodeAgent", "geocode_primary", "geo", depends_on=["nl_parse"], parsed_key="parsed_nl")
        add(
            "time_window",
            "TimeWindowResolverAgent",
            "resolve_time_window",
            "time_window",
            depends_on=["nl_parse", "geo"],
            parsed_key="parsed_nl",
            geo_key="geo",
            default_past_hours=4,
            default_future_hours=2,
        )
        add(
            "normalize",
            "RequestNormalizerAgent",
            "apply_inferred_fields",
            "normalized",
            depends_on=["geo", "time_window"],
            parsed_key="parsed_nl",
            geo_key="geo",
            time_key="time_window",
        )

        # ---- traffic flow (optional) ----
        if flags.flow:
            add("nearest_sensor", "NearestSensorAgent", "match", "nearest_sensor", depends_on=["normalize"])
            add(
                "traffic_flow",
                "TrafficFlowAgent",
                "fetch",
                "sensor_flow",
                depends_on=["nearest_sensor"],
                match_key="nearest_sensor",
                topk=8,
            )

        # ---- satellite (optional) ----
        if flags.satellite:
            add(
                "satellite_image",
                "SatelliteImageAgent",
                "fetch",
                "satellite_image",
                depends_on=["normalize"],
                soft_fail=True,
            )
            add(
                "satellite_image_store",
                "SatelliteImageStoreMySQLAgent",
                "store",
                "satellite_image_store",
                depends_on=["satellite_image"],
                input_key="satellite_image",
                soft_fail=True,
            )

        # ---- context (optional, usually on) ----
        if flags.context:
            add("place", "ReverseGeocodeAgent", "get_place", "place", depends_on=["normalize"])
            add("roads", "RoadContextAgent", "find_nearby_roads", "roads", depends_on=["normalize"], soft_fail=True)
            add("weather", "WeatherAgent", "fetch_weather_hourly", "weather", depends_on=["normalize"])
            add("poi", "POIAgent", "fetch_poi_summary", "poi", depends_on=["normalize"], radius_km=settings.poi_radius_km)
            add("demo", "DemographicsAgent", "compute_density", "demographics", depends_on=["normalize"], radius_km=settings.demo_radius_km)

        # ---- web events (optional) ----
        if flags.web_events:
            # ---- NEWS ----
            add(
                "news_queries",
                "QueryGeneratorAgent",
                "generate",
                "news_queries",
                depends_on=["place", "roads"],
                kind="news",
                place_key="place",
                roads_key="roads",
            )
            add(
                "news_search",
                "WebSearchAgent",
                "search",
                "news_search_results",
                depends_on=["news_queries"],
                queries={"$ref": "news_queries"},
                per_query=settings.web_search.top_k_per_query,
            )
            add(
                "news_extract",
                "NewsExtractorAgent",
                "extract_news_events",
                "news_verified",
                depends_on=["news_search"],
                items_key="news_search_results",
                place_key="place",
                # Inputs for refine.
                queries_key="news_queries",
                per_query=settings.web_search.top_k_per_query,
                enable_refine=True,
                max_refine_iters=1,  # Increase to 2 for stronger but costlier refine.
            )
            add(
                "news_convert",
                "VerifiedEventConverterAgent",
                "convert",
                "news_events",
                depends_on=["news_extract"],
                kind="news",
                input_key="news_verified",
                topk=5,
            )

            # ---- ACCIDENTS ----
            add(
                "acc_queries",
                "QueryGeneratorAgent",
                "generate",
                "accident_queries",
                depends_on=["place", "roads"],
                kind="accidents",
                place_key="place",
                roads_key="roads",
            )
            add(
                "acc_search",
                "WebSearchAgent",
                "search",
                "accident_search_results",
                depends_on=["acc_queries"],
                queries={"$ref": "accident_queries"},
                per_query=settings.web_search.top_k_per_query,
            )
            add(
                "acc_extract",
                "AccidentExtractorAgent",
                "extract_accident_events",
                "acc_verified",
                depends_on=["acc_search"],
                items_key="accident_search_results",
                place_key="place",
                # Inputs for refine.
                queries_key="accident_queries",
                per_query=settings.web_search.top_k_per_query,
                enable_refine=True,
                max_refine_iters=1,
            )
            add(
                "acc_convert",
                "VerifiedEventConverterAgent",
                "convert",
                "accident_events",
                depends_on=["acc_extract"],
                kind="accident",
                input_key="acc_verified",
                topk=5,
            )

        # ---- record (required, always last) ----
        add(
            "record",
            "RecordBuilder",
            "build_dataset_record",
            "dataset_record",
            depends_on=[s.step_id for s in steps],
        )

        return Plan(version="v1", steps=steps, final_outputs=["dataset_record"])
