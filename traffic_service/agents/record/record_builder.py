# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from typing import Any, List

from traffic_service.agents.base import ActionAgent
from traffic_service.core.context import ExecutionContext
from traffic_service.llm.prompts import record_builder_system_prompt
from traffic_service.schemas import (
    DatasetRecord,
    ChatMessage,
    ChatContent,
    Chunks,
    POISummary,
    WeatherHourly,
    DemographicsSummary,
    NewsEvent,
    AccidentEvent,
)


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _poi_lines(poi: POISummary) -> List[str]:
    lines: List[str] = []
    counts = getattr(poi, "counts_by_category", None)
    if isinstance(counts, dict):
        for k, v in counts.items():
            lines.append(f"{v} {k}")
    return lines


def _news_lines(news: List[Any]) -> List[str]:
    out: List[str] = []
    for e in news or []:
        when = _get_field(e, "event_time") or "unknown time"
        loc = _get_field(e, "location_text") or "unknown location"
        cat = _get_field(e, "category") or "news"
        tone_v = _get_field(e, "tone")
        tone = f", tone {tone_v}" if tone_v is not None else ""
        url_v = _get_field(e, "source_url")
        url = f" ({url_v})" if url_v else ""
        summary = _get_field(e, "summary") or ""
        out.append(f"On {when}, {cat} in {loc}: {summary}{tone}.{url}")
    return out


def _accident_lines(accs: List[Any]) -> List[str]:
    out: List[str] = []
    for e in accs or []:
        when = _get_field(e, "event_time") or "unknown time"
        loc = _get_field(e, "location_text") or "unknown location"
        ctype = _get_field(e, "collision_type") or "accident"

        fat_v = _get_field(e, "fatalities")
        inj_v = _get_field(e, "injuries")
        fat = f"{fat_v} fatalities" if fat_v is not None else "unknown fatalities"
        inj = f"{inj_v} injuries" if inj_v is not None else "unknown injuries"

        cond_v = _get_field(e, "conditions")
        cond = f", {cond_v}" if cond_v else ""

        url_v = _get_field(e, "source_url")
        url = f" ({url_v})" if url_v else ""

        summary = _get_field(e, "summary") or ""
        out.append(f"On {when}, a {ctype} occurred near {loc}, resulting in {fat} and {inj}{cond}. {summary}{url}")
    return out


class RecordBuilder(ActionAgent):
    name = "RecordBuilder"

    async def build_dataset_record(self, ctx: ExecutionContext) -> DatasetRecord:
        req = ctx.request

        weather: WeatherHourly | Any = ctx.get("weather", None)
        poi: POISummary | Any = ctx.get("poi", None)
        demo: DemographicsSummary | Any = ctx.get("demographics", None)
        news: List[NewsEvent] | Any = ctx.get("news_events", [])
        accs: List[AccidentEvent] | Any = ctx.get("accident_events", [])

        poi_lines = _poi_lines(poi) if isinstance(poi, POISummary) else []
        news_lines = _news_lines(news) if isinstance(news, list) else []
        accident_lines = _accident_lines(accs) if isinstance(accs, list) else []

        hop_sensor_lines = ctx.get("hop_sensor_lines", None)
        if not hop_sensor_lines:
            sensor_flow = ctx.get("sensor_flow", {})
            if isinstance(sensor_flow, dict):
                hop_sensor_lines = sensor_flow.get("hop_sensor_lines")
        hop_sensor_lines = hop_sensor_lines or []

        sat = ctx.get("satellite_image", None)

        chunks = Chunks(
            POI=poi_lines,
            News=news_lines,
            Accident=accident_lines,
            HopSensor=list(hop_sensor_lines) if isinstance(hop_sensor_lines, list) else [],
            HopBA=[],
        )

        tw = ctx.get("time_window", {}) or {}
        tz_name = tw.get("tz_name") if isinstance(tw, dict) else None
        utc_start = tw.get("utc_start") if isinstance(tw, dict) else None
        utc_end = tw.get("utc_end") if isinstance(tw, dict) else None

        header_parts: List[str] = []
        if getattr(req, "sensor_id", None):
            header_parts.append(f"Sensor ID: {req.sensor_id}")
        header_parts.append(f"Lon: {req.lon}")
        header_parts.append(f"Lat: {req.lat}")

        if getattr(req, "local_start", None) and getattr(req, "local_end", None):
            header_parts.append(f"Local window: {req.local_start} -> {req.local_end}")
        if tz_name:
            header_parts.append(f"TZ: {tz_name}")
        if utc_start and utc_end:
            header_parts.append(f"UTC window: {utc_start} -> {utc_end}")

        if getattr(req, "lanes", None) is not None:
            header_parts.append(f"Lanes: {req.lanes}")
        if getattr(req, "direction", None):
            header_parts.append(f"Direction: {req.direction}")
        if getattr(req, "freeway", None):
            header_parts.append(f"Fwy: {req.freeway}")

        header = "  ".join(header_parts)

        demo_text = ""
        if isinstance(demo, DemographicsSummary):
            demo_text = (
                f"Population density ({demo.radius_km}km): {demo.population_density}; "
                f"Housing density ({demo.radius_km}km): {demo.housing_density}. "
            )

        weather_text = ""
        if isinstance(weather, WeatherHourly):
            weather_text = (
                f"Weather (hourly) {req.local_start} - {req.local_end}: "
                f"Temperature={getattr(weather, 'temperature_c', [])} °C, "
                f"Wind speed={getattr(weather, 'wind_speed_ms', [])} m/s, "
                f"Wind direction={getattr(weather, 'wind_direction_deg', [])}°, "
            )

        flow_text = ""
        if getattr(req, "traffic_flow", None) is not None:
            flow_text += f"traffic flow: {req.traffic_flow}, "
        if getattr(req, "history_average_flow", None) is not None:
            flow_text += f"History average flow: {req.history_average_flow} "

        user_text = (
            f"{header} Given the contexts: [chunk token] [image token] "
            f"{demo_text}{weather_text}{flow_text}"
            f"Please answer the following question: {req.question}"
        )

        user_contents = [ChatContent(type="text", text=user_text)]

        if isinstance(sat, dict):
            img = sat.get("image_ref") or sat.get("data") or sat.get("cache_path") or sat.get("image_id")
            if img:
                user_contents.append(
                    ChatContent(
                        type="image",
                        image=str(img),
                        mime_type=sat.get("mime_type") or "image/png",
                    )
                )

        messages = [
            ChatMessage(role="system", content=[ChatContent(type="text", text=record_builder_system_prompt())]),
            ChatMessage(role="user", content=user_contents),
        ]

        return DatasetRecord(task="reason", messages=messages, chunks=chunks, thinking=[])
