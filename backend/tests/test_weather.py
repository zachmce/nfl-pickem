"""Offline unit tests for the Open-Meteo weather adapter (260710-29v Task 1).

Three layers are exercised, all fully OFFLINE (no network socket, no Redis socket):

* the STATIC :data:`app.services.weather.STADIUMS` table + pure
  :func:`app.services.weather.lookup_stadium` — 32 real NFL stadiums keyed by the
  home-team abbreviation, each with lat/lon floats, a bool ``indoor``, and a name;
  the known dome/retractable set is flagged indoor; lookup is case-insensitive;
* the PURE :func:`app.services.weather.parse_forecast` against a captured Open-Meteo
  ``forecast`` fixture (kickoff-hour indexing) plus inline malformed inputs — it
  must never raise and must return the right dict / ``None`` distinction, degrading
  a single missing metric to ``None`` (never inventing a condition);
* the IMPURE :func:`app.services.weather.fetch_forecast` with ``httpx`` and the
  ``_redis_client`` seam monkeypatched (mirroring ``tests/test_espn_extra.py``) to
  prove cache HIT (no HTTP), cache MISS (HTTP + cache write), best-effort ``None`` on
  a fetch/non-200 failure, and Redis fail-open on the read.

One OPTIONAL live Open-Meteo smoke test is SKIPPED unless ``RUN_WEATHER_LIVE`` is set
(mirrors ``tests/test_espn_extra.py``'s ``RUN_ESPN_LIVE`` gate).

Run with: ``backend/.venv/bin/python -m unittest tests.test_weather -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import httpx

from app.services import weather

_FIXTURE = Path(__file__).parent / "fixtures" / "open_meteo_forecast.json"


def _run(coro):
    return asyncio.run(coro)


def _load_fixture() -> dict:
    return json.loads(_FIXTURE.read_text())


# --------------------------------------------------------------------------- #
# Fake outbound seams (never open a real socket) — mirror tests/test_espn_extra.py.
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _CapturingAsyncClient:
    """A stand-in for ``httpx.AsyncClient`` that records the GET and returns a canned response."""

    last_url: str | None = None
    calls: int = 0
    _response: object = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def get(self, url, *, headers=None):
        type(self).calls += 1
        type(self).last_url = url
        return self._response


class _RaisingAsyncClient:
    """An ``httpx.AsyncClient`` stand-in that FAILS if any HTTP is attempted."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def get(self, url, *, headers=None):
        raise AssertionError("HTTP must not be called on a cache hit")


class _FakeRedis:
    """A minimal async Redis stand-in recording get/set, with a seeded store."""

    def __init__(self, store: dict | None = None) -> None:
        self.store = store or {}
        self.sets: list[tuple[str, str, int | None]] = []

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, *, ex=None):
        self.sets.append((key, value, ex))
        self.store[key] = value

    async def aclose(self):
        return None


def _redis_returns(client: _FakeRedis):
    return mock.patch.object(weather, "_redis_client", lambda: client)


def _redis_raises():
    def _boom():
        raise RuntimeError("redis down")

    return mock.patch.object(weather, "_redis_client", _boom)


# --------------------------------------------------------------------------- #
# Static stadium table + lookup.
# --------------------------------------------------------------------------- #

# The known dome / fixed-roof / retractable-usually-closed venues (per the design
# note + plan). Everything else is an open-air stadium.
_INDOOR_ABBRS = {"ATL", "NO", "DET", "MIN", "LV", "LAR", "LAC", "ARI", "DAL", "HOU", "IND"}


class StadiumTableTests(unittest.TestCase):
    def test_has_exactly_32_rows_all_well_formed(self) -> None:
        self.assertEqual(len(weather.STADIUMS), 32)
        for abbr, stadium in weather.STADIUMS.items():
            self.assertIsInstance(abbr, str)
            self.assertTrue(abbr.strip())
            self.assertIsInstance(stadium.name, str)
            self.assertTrue(stadium.name.strip())
            self.assertIsInstance(stadium.lat, float)
            self.assertIsInstance(stadium.lon, float)
            self.assertIsInstance(stadium.indoor, bool)
            # Sane NFL-region coordinate bounds (continental US).
            self.assertTrue(24.0 < stadium.lat < 49.0, f"{abbr} lat out of range")
            self.assertTrue(-125.0 < stadium.lon < -66.0, f"{abbr} lon out of range")

    def test_dome_set_is_flagged_indoor(self) -> None:
        for abbr in _INDOOR_ABBRS:
            self.assertIn(abbr, weather.STADIUMS)
            self.assertTrue(weather.STADIUMS[abbr].indoor, f"{abbr} should be indoor")

    def test_open_air_stadiums_are_not_indoor(self) -> None:
        for abbr in ("BUF", "GB", "NE", "KC", "CHI"):
            self.assertFalse(weather.STADIUMS[abbr].indoor, f"{abbr} should be outdoor")

    def test_only_the_known_set_is_indoor(self) -> None:
        indoor = {a for a, s in weather.STADIUMS.items() if s.indoor}
        self.assertEqual(indoor, _INDOOR_ABBRS)

    def test_la_teams_share_sofi_and_are_indoor(self) -> None:
        lar = weather.STADIUMS["LAR"]
        lac = weather.STADIUMS["LAC"]
        self.assertTrue(lar.indoor and lac.indoor)
        # Both LA teams share SoFi -> same coordinates + name.
        self.assertEqual((lar.lat, lar.lon), (lac.lat, lac.lon))
        self.assertIn("SoFi", lar.name)
        self.assertIn("SoFi", lac.name)

    def test_lookup_resolves_and_is_case_insensitive(self) -> None:
        self.assertIs(weather.lookup_stadium("BUF"), weather.STADIUMS["BUF"])
        self.assertIs(weather.lookup_stadium("buf"), weather.STADIUMS["BUF"])
        self.assertIs(weather.lookup_stadium("  buf "), weather.STADIUMS["BUF"])

    def test_lookup_unknown_abbr_returns_none(self) -> None:
        self.assertIsNone(weather.lookup_stadium("ZZZ"))
        self.assertIsNone(weather.lookup_stadium(""))


# --------------------------------------------------------------------------- #
# Pure parser — kickoff-hour indexing + defensive degrade.
# --------------------------------------------------------------------------- #


class ParseForecastTests(unittest.TestCase):
    def test_kickoff_hour_selects_the_matching_index_not_index_zero(self) -> None:
        # Kickoff at 14:00 UTC -> index 2 in the fixture (times start at 12:00).
        kickoff = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
        out = weather.parse_forecast(_load_fixture(), kickoff)
        assert out is not None
        # The values must be the index-2 values, NOT index 0 (30.1 / 10.1 / 0.0).
        self.assertEqual(out["temperature_f"], 33.4)
        self.assertEqual(out["wind_mph"], 15.6)
        self.assertEqual(out["precip_in"], 0.05)
        self.assertEqual(out["hour"], "2026-01-05T14:00")

    def test_kickoff_minutes_floor_to_the_hour(self) -> None:
        # 15:47 floors to the 15:00 key -> index 3.
        kickoff = datetime(2026, 1, 5, 15, 47, tzinfo=timezone.utc)
        out = weather.parse_forecast(_load_fixture(), kickoff)
        assert out is not None
        self.assertEqual(out["hour"], "2026-01-05T15:00")
        self.assertEqual(out["temperature_f"], 34.0)
        self.assertEqual(out["wind_mph"], 9.8)

    def test_naive_kickoff_is_treated_as_utc(self) -> None:
        naive = datetime(2026, 1, 5, 14, 0)  # no tzinfo -> assumed UTC
        aware = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
        self.assertEqual(
            weather.parse_forecast(_load_fixture(), naive),
            weather.parse_forecast(_load_fixture(), aware),
        )

    def test_non_utc_kickoff_is_normalized_to_utc(self) -> None:
        # 09:00 at UTC-5 == 14:00 UTC -> index 2.
        from datetime import timedelta

        kickoff = datetime(2026, 1, 5, 9, 0, tzinfo=timezone(timedelta(hours=-5)))
        out = weather.parse_forecast(_load_fixture(), kickoff)
        assert out is not None
        self.assertEqual(out["hour"], "2026-01-05T14:00")
        self.assertEqual(out["temperature_f"], 33.4)

    def test_hour_not_present_returns_none(self) -> None:
        # A kickoff whose hour is not in the fixture -> None (never a neighbor guess).
        kickoff = datetime(2026, 1, 5, 23, 0, tzinfo=timezone.utc)
        self.assertIsNone(weather.parse_forecast(_load_fixture(), kickoff))

    def test_non_dict_payload_returns_none(self) -> None:
        kickoff = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
        for bad in (None, "garbage", 42, ["hourly"]):
            self.assertIsNone(weather.parse_forecast(bad, kickoff))

    def test_missing_or_malformed_hourly_returns_none(self) -> None:
        kickoff = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
        self.assertIsNone(weather.parse_forecast({}, kickoff))
        self.assertIsNone(weather.parse_forecast({"hourly": "nope"}, kickoff))
        self.assertIsNone(weather.parse_forecast({"hourly": {"time": "nope"}}, kickoff))
        self.assertIsNone(weather.parse_forecast({"hourly": {}}, kickoff))

    def test_missing_single_metric_degrades_to_none_but_returns_dict(self) -> None:
        # The hour is present, temperature is present, but wind array is short and
        # precip is non-numeric -> those two degrade to None (never invented); the
        # dict still returns because >=1 metric is present.
        payload = {
            "hourly": {
                "time": ["2026-01-05T14:00"],
                "temperature_2m": [40.0],
                "wind_speed_10m": [],  # short -> no value at index 0
                "precipitation": ["oops"],  # non-numeric -> None
            }
        }
        kickoff = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
        out = weather.parse_forecast(payload, kickoff)
        assert out is not None
        self.assertEqual(out["temperature_f"], 40.0)
        self.assertIsNone(out["wind_mph"])
        self.assertIsNone(out["precip_in"])
        self.assertEqual(out["hour"], "2026-01-05T14:00")

    def test_all_three_metrics_missing_returns_none(self) -> None:
        payload = {
            "hourly": {
                "time": ["2026-01-05T14:00"],
                "temperature_2m": [None],
                "wind_speed_10m": [None],
                "precipitation": [None],
            }
        }
        kickoff = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
        self.assertIsNone(weather.parse_forecast(payload, kickoff))

    def test_bool_is_not_a_valid_metric(self) -> None:
        # A JSON true/false must never be read as a numeric metric.
        payload = {
            "hourly": {
                "time": ["2026-01-05T14:00"],
                "temperature_2m": [True],
                "wind_speed_10m": [False],
                "precipitation": [True],
            }
        }
        kickoff = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
        self.assertIsNone(weather.parse_forecast(payload, kickoff))


# --------------------------------------------------------------------------- #
# Impure fetch (cache + HTTP), fully monkeypatched.
# --------------------------------------------------------------------------- #


class FetchForecastTests(unittest.TestCase):
    def test_cache_hit_returns_payload_without_http(self) -> None:
        payload = _load_fixture()
        lat, lon = 42.77, -78.79
        fake = _FakeRedis({weather._cache_key(lat, lon): json.dumps(payload)})
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _RaisingAsyncClient):
            out = _run(weather.fetch_forecast(lat, lon))
        self.assertEqual(out, payload)  # served from cache
        self.assertEqual(fake.sets, [])  # nothing re-written on a hit

    def test_cache_miss_fetches_and_writes_cache(self) -> None:
        payload = _load_fixture()
        lat, lon = 42.77, -78.79
        fake = _FakeRedis()  # empty -> miss
        _CapturingAsyncClient.calls = 0
        _CapturingAsyncClient.last_url = None
        _CapturingAsyncClient._response = _FakeResponse(200, payload)
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(weather.fetch_forecast(lat, lon))
        self.assertEqual(out, payload)
        self.assertEqual(_CapturingAsyncClient.calls, 1)  # exactly one GET on a miss
        assert _CapturingAsyncClient.last_url is not None
        # The location is carried in the query (lat/lon are NEVER user text — SSRF-safe).
        self.assertIn("latitude=42.77", _CapturingAsyncClient.last_url)
        self.assertIn("longitude=-78.79", _CapturingAsyncClient.last_url)
        # The raw payload was written to the cache under the location key + TTL.
        self.assertEqual(len(fake.sets), 1)
        key, value, ex = fake.sets[0]
        self.assertEqual(key, weather._cache_key(lat, lon))
        self.assertEqual(json.loads(value), payload)
        self.assertEqual(ex, weather.WEATHER_CACHE_TTL_SECONDS)

    def test_http_error_degrades_to_none(self) -> None:
        fake = _FakeRedis()

        class _BoomClient(_CapturingAsyncClient):
            async def get(self, url, *, headers=None):
                raise httpx.ConnectError("boom")

        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _BoomClient):
            out = _run(weather.fetch_forecast(1.0, 2.0))
        self.assertIsNone(out)
        self.assertEqual(fake.sets, [])  # nothing cached on a failed fetch

    def test_non_200_degrades_to_none(self) -> None:
        fake = _FakeRedis()
        _CapturingAsyncClient._response = _FakeResponse(503, {})
        with _redis_returns(fake), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(weather.fetch_forecast(1.0, 2.0))
        self.assertIsNone(out)
        self.assertEqual(fake.sets, [])

    def test_redis_error_still_allows_live_fetch(self) -> None:
        # A Redis outage on the read must FAIL OPEN -> the live fetch still happens.
        payload = _load_fixture()
        _CapturingAsyncClient.calls = 0
        _CapturingAsyncClient._response = _FakeResponse(200, payload)
        with _redis_raises(), mock.patch.object(httpx, "AsyncClient", _CapturingAsyncClient):
            out = _run(weather.fetch_forecast(3.0, 4.0))
        self.assertEqual(out, payload)
        self.assertEqual(_CapturingAsyncClient.calls, 1)


@unittest.skipUnless(
    os.environ.get("RUN_WEATHER_LIVE"),
    "live Open-Meteo smoke test (set RUN_WEATHER_LIVE=1 to enable; performs a real network GET)",
)
class WeatherLiveSmokeTest(unittest.TestCase):
    """OPTIONAL, network-gated smoke test — skipped by default (offline suite)."""

    def test_fetch_forecast_returns_a_dict(self) -> None:
        buf = weather.STADIUMS["BUF"]
        out = _run(weather.fetch_forecast(buf.lat, buf.lon))
        self.assertIsInstance(out, dict)


if __name__ == "__main__":
    unittest.main()
