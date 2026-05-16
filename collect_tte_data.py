#!/usr/bin/env python3
"""Collect travel‑time estimation training data for Hsinchu City.

Produces a CSV with columns:
    start_lat, start_lon, end_lat, end_lon,
    time_slot, hour, weekday,
    graphhopper_seconds, google_maps_seconds,
    distance_meters

Requirements:
    pip install requests googlemaps pandas
"""

import csv
import os
import math
import random
import time
from datetime import datetime, timedelta

import googlemaps
import requests

# --------------------------------------------------------------------------- #
#  Configuration                                                              #
# --------------------------------------------------------------------------- #
GRAPHOPPER_BASE = "http://localhost:8989"

# Bounding‑box of Hsinchu City (approximate)
BBOX = {
    "min_lat": 24.5,
    "max_lat": 25.0,
    "min_lon": 120.7,
    "max_lon": 121.2,
}

NUM_PAIRS = 100
MIN_STRAIGHT_LINE_M = 500
MAX_STRAIGHT_LINE_M = 25_000

# Time slots: (label, weekday, hour)
# weekday: Monday=0, Saturday=5
TIME_SLOTS = [
    ("周一早峰", 0, 8),
    ("周一晚峰", 0, 18),
    ("周一下午", 0, 14),
    ("周六", 5, 10),
]

OUTPUT_FILE = "tte_data.csv"

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
if not GOOGLE_MAPS_API_KEY:
    raise EnvironmentError("Environment variable GOOGLE_MAPS_API_KEY not set")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Rate‑limiting: 10 requests/second => 100 ms between calls
GOOGLE_RATE_LIMIT_S = 0.1
MAX_RETRIES = 3


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def haversine_m(lat1: float, lon1: float,
                lat2: float, lon2: float) -> float:
    """Return straight‑line distance in metres (Haversine formula)."""
    R = 6371000                     # Earth radius in metres
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def next_weekday_from_today(target_weekday: int, target_hour: int) -> datetime:
    """Return the nearest future ``target_weekday`` at ``target_hour``:00."""
    today = datetime.now()
    days_ahead = target_weekday - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7                     # move to next week
    future_date = today + timedelta(days=days_ahead)
    return future_date.replace(hour=target_hour, minute=0, second=0, microsecond=0)


def generate_od_pairs(num_pairs: int,
                      min_dist: float,
                      max_dist: float,
                      bbox: dict) -> list[tuple[float, float, float, float]]:
    """Return list of ``(start_lat, start_lon, end_lat, end_lon)``."""
    pairs: list = []
    while len(pairs) < num_pairs:
        slat = random.uniform(bbox["min_lat"], bbox["max_lat"])
        slon = random.uniform(bbox["min_lon"], bbox["max_lon"])
        elat = random.uniform(bbox["min_lat"], bbox["max_lat"])
        elon = random.uniform(bbox["min_lon"], bbox["max_lon"])
        d = haversine_m(slat, slon, elat, elon)
        if min_dist <= d <= max_dist:
            pairs.append((slat, slon, elat, elon))
    return pairs


def graphhopper_route(lat1: float, lon1: float,
                      lat2: float, lon2: float) -> tuple[float | None,
                                                         float | None]:
    """Return ``(travel_seconds, distance_meters)`` from local GraphHopper."""
    url = f"{GRAPHOPPER_BASE}/route"
    params = {
        "point": [f"{lat1},{lon1}", f"{lat2},{lon2}"],
        "profile": "car",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        path = data.get("paths", [])
        if not path:
            return None, None
        time_ms = path[0].get("time")
        dist_m = path[0].get("distance")
        if time_ms is None:
            return None, dist_m
        return time_ms / 1000, dist_m
    except Exception:
        return None, None


def google_duration(gmaps_client: googlemaps.Client,
                    origin: tuple[float, float],
                    destination: tuple[float, float],
                    departure_time: datetime) -> float | None:
    """Return ``duration_in_traffic`` (seconds) from Google Maps API."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            result = gmaps_client.distance_matrix(
                origins=origin,
                destinations=destination,
                departure_time=departure_time,
                mode="driving",
            )
            # Wait after every call (rate‑limit enforcement)
            time.sleep(GOOGLE_RATE_LIMIT_S)

            elements = (result.get("rows", [{}])[0]
                        .get("elements", [{}]))
            elem = elements[0]
            if elem.get("status") != "OK":
                err_msg = elem.get("status", "unknown status")
                raise RuntimeError(f"Google API element status: {err_msg}")
            dur = (elem.get("duration_in_traffic", {})
                   .get("value"))
            return float(dur) if dur is not None else None
        except Exception as exc:
            last_err = exc
            if attempt < MAX_RETRIES - 1:
                backoff = 2 ** attempt
                time.sleep(backoff + GOOGLE_RATE_LIMIT_S)
            continue
    raise RuntimeError(f"Google API failed after {MAX_RETRIES} attempts") from last_err


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main() -> None:
    # 1. initialise Google Maps client
    gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

    # 2. generate OD pairs
    print("Generating OD pairs ...")
    pairs = generate_od_pairs(NUM_PAIRS, MIN_STRAIGHT_LINE_M,
                              MAX_STRAIGHT_LINE_M, BBOX)
    print(f"  {len(pairs)} pairs created.")

    # 3. open / create output file
    file_exists = (os.path.isfile(OUTPUT_FILE) and
                   os.path.getsize(OUTPUT_FILE) > 0)
    outfile = open(OUTPUT_FILE, "a", newline="")
    writer = csv.DictWriter(outfile, fieldnames=[
        "start_lat", "start_lon", "end_lat", "end_lon",
        "time_slot", "hour", "weekday",
        "graphhopper_seconds", "google_maps_seconds",
        "distance_meters",
    ])
    if not file_exists:
        writer.writeheader()
        outfile.flush()

    # 4. process each pair
    for idx, (slat, slon, elat, elon) in enumerate(pairs, 1):
        dist_m = haversine_m(slat, slon, elat, elon)

        for slot_label, weekday, hour in TIME_SLOTS:
            departure_dt = next_weekday_from_today(weekday, hour)
            dep_timestamp = int(departure_dt.timestamp())

            print(f"  [{idx}/{len(pairs)}] "
                  f"{slat:.4f},{slon:.4f} -> {elat:.4f},{elon:.4f}  "
                  f"slot={slot_label}  dep={departure_dt.isoformat()}")

            # (a) GraphHopper
            gh_sec, gh_dist = graphhopper_route(slat, slon, elat, elon)
            # (b) Google Maps
            try:
                gmaps_sec = google_duration(gmaps,
                                            (slat, slon),
                                            (elat, elon),
                                            departure_dt)
            except RuntimeError as e:
                print(f"    WARNING Google Maps error: {e}")
                gmaps_sec = None

            row = {
                "start_lat": slat,
                "start_lon": slon,
                "end_lat": elat,
                "end_lon": elon,
                "time_slot": slot_label,
                "hour": hour,
                "weekday": weekday,
                "graphhopper_seconds": gh_sec,
                "google_maps_seconds": gmaps_sec,
                "distance_meters": gh_dist if gh_dist is not None else dist_m,
            }
            writer.writerow(row)
            outfile.flush()

    outfile.close()
    print(f"\nDone. Output written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
