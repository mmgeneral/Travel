#!/usr/bin/env python3
"""
Travel Time Estimation abstraction layer.

Provides:
    - TravelTimeResult (frozen dataclass)
    - TravelTimeProvider (ABC)
    - GraphHopperProvider (car/bike/foot with optional XGBoost correction)
    - OTPProvider (transit via OpenTripPlanner REST API)
    - TravelTimeRouter (dispatches by mode)

Usage example (see bottom of file).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Data structures                                                             #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TravelTimeResult:
    """Result of a travel‑time estimation request."""

    estimated_seconds: int
    distance_meters: int
    mode: str                         # raw mode passed by caller
    confidence: str                   # "high" / "medium" / "low"
    legs: List[Dict[str, Any]] = field(default_factory=list)
    provider: str = ""               # identifies which backend was used


# --------------------------------------------------------------------------- #
#  Abstract base class                                                         #
# --------------------------------------------------------------------------- #

class TravelTimeProvider(ABC):
    """Abstract travel‑time estimator."""

    @abstractmethod
    async def estimate(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        mode: str,
        departure_time: Optional[datetime] = None,
    ) -> TravelTimeResult:
        """Return an estimated travel time result."""
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Lightweight health‑check for the provider backend."""
        ...

    async def next_arrivals(
        self,
        stop_id: str,
        route_id: str,
        departure_time: Optional[datetime] = None,
    ) -> list[dict]:
        """Return predicted arrivals for a stop/route (reserved for TDX)."""
        raise NotImplementedError("next_arrivals is not implemented yet")


# --------------------------------------------------------------------------- #
#  GraphHopperProvider (car / bike / foot)                                     #
# --------------------------------------------------------------------------- #

_PROFILE_MAP: dict[str, str] = {
    "car": "car",
    "bike": "bike",
    "foot": "foot",
}


class GraphHopperProvider(TravelTimeProvider):
    """Travel‑time from local GraphHopper instance, optionally corrected by
    a pre‑trained XGBoost model."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str = "http://localhost:8989",
        model: Any = None,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._model = model  # might be None

    async def is_available(self) -> bool:
        try:
            resp = await self._client.get(f"{self._base_url}/route", params={"point": ["0,0", "0,0"], "profile": "car"}, timeout=3)
            resp.raise_for_status()
            return True
        except Exception:
            return False

    async def estimate(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        mode: str,
        departure_time: Optional[datetime] = None,
    ) -> TravelTimeResult:
        # 1. resolve profile
        profile = _PROFILE_MAP.get(mode)
        if profile is None:
            raise ValueError(f"Unsupported mode '{mode}' for GraphHopperProvider")
        # 2. call GraphHopper
        params = {
            "point": [f"{start[0]},{start[1]}", f"{end[0]},{end[1]}"],
            "profile": profile,
        }
        resp = await self._client.get(f"{self._base_url}/route", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        paths = data.get("paths", [])
        if not paths:
            raise RuntimeError("GraphHopper returned no paths")

        path = paths[0]
        time_ms: int = path.get("time", 0)
        distance_m: float = path.get("distance", 0.0)

        estimated_seconds = int(time_ms / 1000)
        distance_meters = int(distance_m)

        # 3. XGBoost correction (car only, when model is present)
        model_corrected = False
        if self._model is not None and mode == "car":
            now = departure_time if departure_time is not None else datetime.now(timezone.utc)
            hour = now.hour
            weekday = now.weekday()
            is_peak = 1 if hour in (8, 9, 17, 18) else 0
            is_weekend = 1 if weekday >= 5 else 0
            features = pd.DataFrame(
                [[estimated_seconds, distance_meters, hour, weekday, is_peak, is_weekend]],
                columns=[
                    "graphhopper_seconds",
                    "distance_meters",
                    "hour",
                    "weekday",
                    "is_peak",
                    "is_weekend",
                ],
            )
            pred_seconds = float(self._model.predict(features)[0])
            # only apply correction if prediction is reasonable (>0)
            if pred_seconds > 0:
                estimated_seconds = int(round(pred_seconds))
                model_corrected = True

        provider = "graphhopper+xgboost" if model_corrected else "graphhopper"

        # 4. confidence based on distance
        if distance_meters < 5000:
            confidence = "high"
        elif distance_meters < 20000:
            confidence = "medium"
        else:
            confidence = "low"

        return TravelTimeResult(
            estimated_seconds=estimated_seconds,
            distance_meters=distance_meters,
            mode=mode,
            confidence=confidence,
            legs=[],
            provider=provider,
        )


# --------------------------------------------------------------------------- #
#  OTPProvider (transit)                                                       #
# --------------------------------------------------------------------------- #

class OTPProvider(TravelTimeProvider):
    """Travel‑time from local OpenTripPlanner REST API (transit)."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str = "http://localhost:8080",
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._plan_url = f"{self._base_url}/otp/routers/default/plan"

    async def is_available(self) -> bool:
        # A simple status request (the plan endpoint returns 400 without params)
        try:
            resp = await self._client.get(self._plan_url, params={"fromPlace": "0,0", "toPlace": "0,0"}, timeout=3)
            # 400 is acceptable (missing params) – 200 is fine
            return resp.status_code in (200, 400, 202)
        except Exception:
            return False

    async def estimate(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        mode: str,
        departure_time: Optional[datetime] = None,
    ) -> TravelTimeResult:
        if mode != "transit":
            raise ValueError("OTPProvider only supports mode='transit'")

        # 1. prepare OTP date / time parameters
        dt = departure_time if departure_time is not None else datetime.now(timezone.utc)
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H:%M:%S")

        params = {
            "fromPlace": f"{start[0]},{start[1]}",
            "toPlace": f"{end[0]},{end[1]}",
            "date": date_str,
            "time": time_str,
            "mode": "TRANSIT,WALK",
            "arriveBy": "false",
            "numItineraries": 1,
        }
        resp = await self._client.get(self._plan_url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        plan = data.get("plan", {})
        itineraries = plan.get("itineraries", [])
        if not itineraries:
            raise RuntimeError("OTP returned no itineraries")

        itinerary = itineraries[0]

        # 2. extract legs
        legs_raw: list[dict] = itinerary.get("legs", [])
        legs: list[dict] = []
        total_seconds = 0
        total_distance = 0.0
        for leg in legs_raw:
            leg_mode = leg.get("mode", "WALK")
            leg_duration = leg.get("duration", 0)  # seconds
            total_seconds += leg_duration

            distance = leg.get("distance", 0.0)
            total_distance += distance

            from_stop = leg.get("from", {})
            to_stop = leg.get("to", {})
            legs.append(
                {
                    "mode": leg_mode,
                    "duration": leg_duration,
                    "from_stop": from_stop.get("name"),
                    "to_stop": to_stop.get("name"),
                }
            )

        distance_meters = int(total_distance)

        # 3. confidence (simplified)
        if distance_meters < 5000:
            confidence = "high"
        elif distance_meters < 20000:
            confidence = "medium"
        else:
            confidence = "low"

        return TravelTimeResult(
            estimated_seconds=int(total_seconds),
            distance_meters=distance_meters,
            mode=mode,
            confidence=confidence,
            legs=legs,
            provider="otp",
        )


# --------------------------------------------------------------------------- #
#  Router                                                                      #
# --------------------------------------------------------------------------- #

_TW_LAT_RANGE = (21.9, 25.3)
_TW_LON_RANGE = (120.0, 122.0)


def _is_taiwan_lat_lon(lat: float, lon: float) -> bool:
    return (_TW_LAT_RANGE[0] <= lat <= _TW_LAT_RANGE[1]
            and _TW_LON_RANGE[0] <= lon <= _TW_LON_RANGE[1])


class TravelTimeRouter:
    """Dispatcher that selects the appropriate provider based on travel mode."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        tte_model: Any = None,
        graphhopper_base: str = "http://localhost:8989",
        otp_base: str = "http://localhost:8080",
    ) -> None:
        self._client = client
        self._graphhopper_provider = GraphHopperProvider(
            client=client, base_url=graphhopper_base, model=tte_model,
        )
        self._otp_provider = OTPProvider(client=client, base_url=otp_base)

    async def estimate(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        mode: str,
        departure_time: Optional[datetime] = None,
    ) -> TravelTimeResult:
        if mode in ("car", "bike", "foot"):
            provider = self._graphhopper_provider
        elif mode == "transit":
            # Is the journey wholly within Taiwan?
            if (_is_taiwan_lat_lon(start[0], start[1])
                    and _is_taiwan_lat_lon(end[0], end[1])):
                raise NotImplementedError("TDXProvider for Taiwan transit is coming soon")
            provider = self._otp_provider
        else:
            raise ValueError(f"Unknown travel mode '{mode}'")

        if not await provider.is_available():
            from fastapi import HTTPException
            raise HTTPException(status_code=503, detail=f"Provider for mode '{mode}' is unavailable")

        return await provider.estimate(start, end, mode, departure_time)


# --------------------------------------------------------------------------- #
#  Usage example (pseudocode – not runnable)                                    #
# --------------------------------------------------------------------------- #
"""
# Example early in the FastAPI app startup:

import httpx
client = httpx.AsyncClient()
tte_model = joblib.load("tte_model.pkl")
router = TravelTimeRouter(client=client, tte_model=tte_model)

# In a request handler:
async def handler():
    result = await router.estimate(
        start=(24.8, 121.0),
        end=(24.5, 120.9),
        mode="car",
        departure_time=datetime.now(),
    )
    print(result.estimated_seconds)

# For transit outside Taiwan:
result = await router.estimate(
    start=(40.71, -74.00),
    end=(40.73, -73.99),
    mode="transit",
)
print(result.estimated_seconds)
"""
