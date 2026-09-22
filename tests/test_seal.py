"""Behaviour tests for source sealing and the tail drain (finite sessions)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("SENSOR_DB_PATH", db_path)
    from app import database, main

    main.app.state.db_path = db_path
    database.init_db(db_path)
    with TestClient(main.app) as c:
        yield c


def create_session(client, sources=("a", "b"), lateness=10.0, session_id=None):
    body = {
        "source_ids": list(sources),
        "allowed_lateness_seconds": lateness,
    }
    if session_id:
        body["session_id"] = session_id
    r = client.post("/sessions", json=body)
    assert r.status_code == 201, r.text
    return r.json()["session_id"]


def report(client, sid, readings):
    return client.post(f"/sessions/{sid}/readings", json={"readings": readings})


def seal(client, sid, source_id, final_observed_at):
    return client.post(
        f"/sessions/{sid}/sources/{source_id}/seal",
        json={"final_observed_at": final_observed_at},
    )


def published_times(client, sid):
    out = []
    cursor = None
    while True:
        params = {"limit": 1000}
        if cursor:
            params["cursor"] = cursor
        page = client.get(f"/sessions/{sid}/stream", params=params).json()
        out.extend(page["readings"])
        cursor = page["next_cursor"]
        if not cursor:
            return out


def session_info(client, sid):
    r = client.get(f"/sessions/{sid}")
    assert r.status_code == 200, r.text
    return r.json()


class TestSealBasics:
    def test_seal_one_source_and_session_status(self, client):
        sid = create_session(client)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:10Z"},
        ])
        r = seal(client, sid, "a", "2026-01-01T00:00:20Z")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["source_id"] == "a"
        assert body["final_observed_at"] == "2026-01-01T00:00:20Z"
        assert body["already_sealed"] is False
        assert body["sealed_sources"] == 1
        assert body["total_sources"] == 2
        assert body["all_sources_sealed"] is False
        assert body["closed"] is False
        assert body["published_now"] == 0

        info = session_info(client, sid)
        assert info["closed"] is False
        assert info["closed_at"] is None
        by_source = {s["source_id"]: s for s in info["sources"]}
        assert by_source["a"]["final_observed_at"] == "2026-01-01T00:00:20Z"
        assert by_source["a"]["max_observed_at"] == "2026-01-01T00:00:10Z"
        assert by_source["b"]["final_observed_at"] is None

    def test_seal_same_value_retry_is_idempotent(self, client):
        sid = create_session(client)
        first = seal(client, sid, "a", "2026-01-01T00:00:20Z").json()
        assert first["already_sealed"] is False
        second = seal(client, sid, "a", "2026-01-01T00:00:20Z")
        assert second.status_code == 200
        body = second.json()
        assert body["already_sealed"] is True
        assert body["sealed_sources"] == 1
        # Equivalent instant written differently is still the same value.
        third = seal(client, sid, "a", "2026-01-01T02:00:20+02:00").json()
        assert third["already_sealed"] is True

    def test_seal_different_value_conflicts(self, client):
        sid = create_session(client)
        seal(client, sid, "a", "2026-01-01T00:00:20Z")
        r = seal(client, sid, "a", "2026-01-01T00:00:30Z")
        assert r.status_code == 409
        # ... and the original value is kept.
        info = session_info(client, sid)
        by_source = {s["source_id"]: s for s in info["sources"]}
        assert by_source["a"]["final_observed_at"] == "2026-01-01T00:00:20Z"

    def test_seal_earlier_than_max_observed_rejected(self, client):
        sid = create_session(client)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:10Z"},
        ])
        r = seal(client, sid, "a", "2026-01-01T00:00:09Z")
        assert r.status_code == 400
        assert "earlier" in r.json()["detail"]
        # Equal to the max seen event time is allowed (boundary).
        ok = seal(client, sid, "a", "2026-01-01T00:00:10Z")
        assert ok.status_code == 200, ok.text

    def test_seal_source_that_never_reported(self, client):
        sid = create_session(client)
        r = seal(client, sid, "a", "2026-01-01T00:00:00Z")
        assert r.status_code == 200, r.text

    def test_seal_unknown_source_and_session(self, client):
        sid = create_session(client)
        assert seal(client, sid, "zzz", "2026-01-01T00:00:00Z").status_code == 400
        assert seal(client, "nope", "a", "2026-01-01T00:00:00Z").status_code == 404

    def test_seal_bad_timestamp_400(self, client):
        sid = create_session(client)
        assert seal(client, sid, "a", "not-a-time").status_code == 400
        r = client.post(f"/sessions/{sid}/sources/a/seal", json={})
        assert r.status_code == 422


class TestSealIngest:
    def test_new_reading_on_sealed_source_rejected(self, client):
        sid = create_session(client)
        seal(client, sid, "a", "2026-01-01T00:00:20Z")
        r = report(client, sid, [
            {"source_id": "a", "seq": 5, "observed_at": "2026-01-01T00:00:15Z"},
        ])
        assert r.status_code == 409
        # Other sources are unaffected.
        ok = report(client, sid, [
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:15Z"},
        ])
        assert ok.status_code == 200, ok.text
        assert ok.json()["received"] == 1

    def test_exact_duplicate_on_sealed_source_still_counts(self, client):
        sid = create_session(client)
        reading = {"source_id": "a", "seq": 1,
                   "observed_at": "2026-01-01T00:00:05Z", "value": {"k": 1}}
        report(client, sid, [dict(reading)])
        seal(client, sid, "a", "2026-01-01T00:00:20Z")
        r = report(client, sid, [dict(reading)])
        assert r.status_code == 200, r.text
        assert r.json()["duplicates"] == 1
        assert r.json()["received"] == 1

    def test_same_key_different_payload_on_sealed_source_conflicts(self, client):
        sid = create_session(client)
        report(client, sid, [
            {"source_id": "a", "seq": 1,
             "observed_at": "2026-01-01T00:00:05Z", "value": "v1"},
        ])
        seal(client, sid, "a", "2026-01-01T00:00:20Z")
        r = report(client, sid, [
            {"source_id": "a", "seq": 1,
             "observed_at": "2026-01-01T00:00:05Z", "value": "DIFFERENT"},
        ])
        assert r.status_code == 409

    def test_watermark_semantics_unchanged_while_sealing(self, client):
        sid = create_session(client, lateness=0.0)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:10Z"},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:10Z"},
        ])
        # Sealing a with a far-future final time must not move the watermark
        # nor publish anything by itself.
        r = seal(client, sid, "a", "2026-01-01T01:00:00Z").json()
        assert r["published_now"] == 0
        info = session_info(client, sid)
        assert info["published_watermark"] == "2026-01-01T00:00:10Z"
        assert info["effective_watermark"] == "2026-01-01T00:00:10Z"
        # b keeps reporting, but the watermark is still pinned by a's frozen
        # max event time (min of per-source maxes, unchanged by the seal):
        # nothing new publishes and nothing is quarantined.
        r = report(client, sid, [
            {"source_id": "b", "seq": 2, "observed_at": "2026-01-01T00:00:20Z"},
        ]).json()
        assert r["published_watermark"] == "2026-01-01T00:00:10Z"
        assert r["published_now"] == 0
        assert r["buffered"] == 3
        # Sealing the last source drains the buffer in merge order.
        r = seal(client, sid, "b", "2026-01-01T00:00:20Z").json()
        assert r["closed"] is True
        assert r["published_now"] == 3
        assert r["published_watermark"] == "2026-01-01T01:00:00Z"
        rows = published_times(client, sid)
        assert [(x["source_id"], x["seq"]) for x in rows] == [
            ("a", 1), ("b", 1), ("b", 2)
        ]


class TestTailDrain:
    def test_last_seal_drains_buffer_in_order_and_closes(self, client):
        sid = create_session(client, sources=["a", "b"], lateness=0.0)
        report(client, sid, [
            {"source_id": "b", "seq": 2, "observed_at": "2026-01-01T00:00:05Z"},
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:05Z"},
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T00:00:10Z"},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:01Z"},
        ])
        # Watermark is 00:00:05 (slowest max, lateness 0): only b@:01 is
        # strictly before it and published by the batch; the rest buffers.
        assert [(x["source_id"], x["seq"]) for x in published_times(client, sid)] == [
            ("b", 1)
        ]

        seal(client, sid, "a", "2026-01-01T00:00:10Z")
        r = seal(client, sid, "b", "2026-01-01T00:00:05Z")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["all_sources_sealed"] is True
        assert body["closed"] is True
        assert body["published_now"] == 3
        assert body["buffered"] == 0
        # Final watermark: max of the per-source final observed times.
        assert body["published_watermark"] == "2026-01-01T00:00:10Z"

        rows = published_times(client, sid)
        assert [(x["source_id"], x["seq"]) for x in rows] == [
            ("b", 1), ("a", 1), ("b", 2), ("a", 2)
        ]
        info = session_info(client, sid)
        assert info["closed"] is True
        assert info["closed_at"] is not None
        assert info["published_watermark"] == "2026-01-01T00:00:10Z"
        assert info["effective_watermark"] == "2026-01-01T00:00:10Z"

    def test_drain_with_empty_buffer_closes_without_publish(self, client):
        sid = create_session(client, sources=["a"], lateness=0.0)
        r = seal(client, sid, "a", "2026-01-01T00:00:00Z").json()
        assert r["closed"] is True
        assert r["published_now"] == 0
        assert r["published_total"] == 0

    def test_quarantined_readings_stay_quarantined_after_drain(self, client):
        sid = create_session(client, lateness=0.0)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:10Z"},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:10Z"},
        ])
        # Late reading behind the published watermark -> quarantine.
        r = report(client, sid, [
            {"source_id": "a", "seq": 9, "observed_at": "2026-01-01T00:00:01Z"},
        ]).json()
        assert r["quarantined"] == 1

        seal(client, sid, "a", "2026-01-01T00:00:10Z")
        seal(client, sid, "b", "2026-01-01T00:00:10Z")
        # The drain publishes only buffered readings; the quarantined one
        # never enters the published stream.
        rows = published_times(client, sid)
        assert [(x["source_id"], x["seq"]) for x in rows] == [("a", 1), ("b", 1)]
        q = client.get(f"/sessions/{sid}/quarantine").json()["readings"]
        assert len(q) == 1 and q[0]["seq"] == 9

    def test_closed_session_rejects_new_keys_but_counts_duplicates(self, client):
        sid = create_session(client, sources=["a", "b"], lateness=0.0)
        reading = {"source_id": "a", "seq": 1,
                   "observed_at": "2026-01-01T00:00:10Z", "value": 1}
        report(client, sid, [dict(reading),
                             {"source_id": "b", "seq": 1,
                              "observed_at": "2026-01-01T00:00:10Z", "value": 2}])
        seal(client, sid, "a", "2026-01-01T00:00:10Z")
        seal(client, sid, "b", "2026-01-01T00:00:10Z")
        assert session_info(client, sid)["closed"] is True

        # New keys are rejected, for sealed sources on any source.
        assert report(client, sid, [
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T00:00:11Z"},
        ]).status_code == 409
        assert report(client, sid, [
            {"source_id": "b", "seq": 2, "observed_at": "2026-01-01T00:00:11Z"},
        ]).status_code == 409
        # Exact retries of existing readings still count as duplicates.
        r = report(client, sid, [dict(reading)]).json()
        assert r["duplicates"] == 1
        # Seals stay idempotent after close; a different value still conflicts.
        assert seal(client, sid, "a", "2026-01-01T00:00:10Z").json()[
            "already_sealed"] is True
        assert seal(client, sid, "a", "2026-01-01T00:00:11Z").status_code == 409

    def test_drain_advances_threshold_alerts(self, client):
        sid = create_session(client, lateness=0.0)
        client.post(f"/sessions/{sid}/alert-rules", json={
            "rule_id": "high-a",
            "source_id": "a",
            "direction": "above",
            "enter_threshold": 10.0,
            "recover_threshold": 5.0,
            "enter_consecutive": 2,
            "recover_consecutive": 1,
        })
        # Two breaching readings stay buffered (on the watermark boundary).
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:01Z", "value": 11},
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T00:00:02Z", "value": 12},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:02Z", "value": 0},
        ])
        assert client.get(f"/sessions/{sid}/alerts/current").json()["alerts"] == []

        seal(client, sid, "a", "2026-01-01T00:00:02Z")
        seal(client, sid, "b", "2026-01-01T00:00:02Z")
        # The drain published both breaching readings inside the seal
        # transaction, so the state machine entered and stays queryable.
        current = client.get(f"/sessions/{sid}/alerts/current").json()["alerts"]
        assert [a["rule_id"] for a in current] == ["high-a"]
        assert current[0]["entered"]["seq"] == 2
        events = client.get(f"/sessions/{sid}/alerts/history").json()["events"]
        assert [e["event_type"] for e in events] == ["entered"]

    def test_drain_advances_correlated_rules(self, client):
        sid = create_session(client, lateness=0.0)
        client.post(f"/sessions/{sid}/correlated-rules", json={
            "rule_id": "both-high",
            "conditions": [
                {"source_id": "a", "direction": "above", "threshold": 10.0,
                 "window_seconds": 30},
                {"source_id": "b", "direction": "above", "threshold": 10.0,
                 "window_seconds": 30},
            ],
        })
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:01Z", "value": 11},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:01Z", "value": 12},
        ])
        assert client.get(
            f"/sessions/{sid}/correlated-alerts/current").json()["alerts"] == []

        seal(client, sid, "a", "2026-01-01T00:00:01Z")
        seal(client, sid, "b", "2026-01-01T00:00:01Z")
        current = client.get(
            f"/sessions/{sid}/correlated-alerts/current").json()["alerts"]
        assert [a["rule_id"] for a in current] == ["both-high"]
        assert current[0]["correlated_entered"]["completed_by"] == {
            "source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:01Z",
        }

    def test_rule_created_after_close_replays_drained_stream(self, client):
        sid = create_session(client, lateness=0.0)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:01Z", "value": 11},
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T00:00:02Z", "value": 12},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:02Z", "value": 0},
        ])
        seal(client, sid, "a", "2026-01-01T00:00:02Z")
        seal(client, sid, "b", "2026-01-01T00:00:02Z")
        r = client.post(f"/sessions/{sid}/alert-rules", json={
            "source_id": "a",
            "direction": "above",
            "enter_threshold": 10.0,
            "recover_threshold": 5.0,
            "enter_consecutive": 2,
            "recover_consecutive": 1,
        })
        assert r.status_code == 201, r.text
        assert r.json()["state"] == "active"


class TestSealMigration:
    def test_v1_database_is_upgraded_in_place(self, tmp_path, monkeypatch):
        import sqlite3

        from app import database
        from app.database import utc_now
        from app.time_utils import store_event_time

        db_path = str(tmp_path / "legacy.db")
        # A database as the previous release (schema v1) left it: base schema
        # plus migration 1, with live data.
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(database.SCHEMA_V0)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            database._migration_1(conn, store_event_time(utc_now()))
            now = store_event_time(utc_now())
            conn.execute(
                "INSERT INTO sessions (session_id, source_ids,"
                " allowed_lateness_seconds, created_at) VALUES (?, ?, 0, ?)",
                ("legacy-s", '["a","b"]', now),
            )
            conn.execute(
                "INSERT INTO session_sources (session_id, source_id,"
                " max_observed_at, first_seen_at) VALUES ('legacy-s', 'a', NULL, ?),"
                " ('legacy-s', 'b', NULL, ?)",
                (now, now),
            )
            conn.commit()
        finally:
            conn.close()

        database.init_db(db_path)
        conn = sqlite3.connect(db_path)
        try:
            versions = [r[0] for r in conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version")]
            assert versions == [1, 2]
            session_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(sessions)")
            }
            assert "closed_at" in session_cols
            source_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(session_sources)")
            }
            assert "final_observed_at" in source_cols
            # Existing rows are preserved with NULL seal/close markers.
            row = conn.execute(
                "SELECT closed_at FROM sessions WHERE session_id = 'legacy-s'"
            ).fetchone()
            assert row[0] is None
        finally:
            conn.close()

        # Re-running init is a no-op.
        database.init_db(db_path)
        conn = sqlite3.connect(db_path)
        try:
            assert conn.execute(
                "SELECT COUNT(*) FROM schema_migrations WHERE version = 2"
            ).fetchone()[0] == 1
        finally:
            conn.close()

        # The upgraded database serves seal traffic end to end.
        monkeypatch.setenv("SENSOR_DB_PATH", db_path)
        from app import main

        main.app.state.db_path = db_path
        with TestClient(main.app) as c:
            r = c.post("/sessions/legacy-s/sources/a/seal",
                       json={"final_observed_at": "2026-01-01T00:00:00Z"})
            assert r.status_code == 200, r.text
            info = c.get("/sessions/legacy-s").json()
            assert info["closed"] is False
            by_source = {s["source_id"]: s for s in info["sources"]}
            assert by_source["a"]["final_observed_at"] == "2026-01-01T00:00:00Z"
