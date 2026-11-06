"""Conversation splitter, SQLite segments, and checkpoint resume."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from common.database import init_db
from importer.import_service import run_import
from segmentation.service import SegmentationConflictError, run_segmentation
from segmentation.splitter import MessageRef, SegmentParams, Segmenter
from tests.conftest import FIXTURE, ROOT
from tests.support import insert_message, seed_messages


def _refs(specs: list[dict]) -> list[MessageRef]:
    refs = []
    for index, spec in enumerate(specs):
        refs.append(
            MessageRef(
                message_id=spec["message_id"],
                conversation_id=spec.get("conversation_id", "conv"),
                timestamp=spec.get("timestamp"),
                source_position=spec.get("source_position", index),
            )
        )
    return refs


def _close_all(messages: list[MessageRef], params: SegmentParams):
    segmenter = Segmenter(params)
    closed = []
    for message in messages:
        closed.extend(segmenter.push(message))
    final = segmenter.finish()
    if final is not None:
        closed.append(final)
    return closed


def _ids(segment) -> list[str]:
    return [item.message_id for item in segment.messages]


def test_params_reject_overlap_that_fills_the_segment():
    with pytest.raises(ValueError):
        SegmentParams(max_messages=4, overlap_messages=4)


def test_size_split_overlap_covers_each_message_once():
    params = SegmentParams(gap_seconds=1800, max_messages=10, overlap_messages=3)
    messages = _refs(
        [{"message_id": f"m{index}", "timestamp": index} for index in range(61)]
    )
    closed = _close_all(messages, params)
    primary: list[str] = []
    for previous, segment in zip([None, *closed], closed):
        assert len(segment.messages) <= 10
        assert segment.target_message_count == len(segment.messages) - segment.overlap_count
        primary.extend(item.message_id for item in segment.primary_messages)
        if segment.opened_by == "max_messages":
            assert previous is not None
            assert _ids(segment)[: segment.overlap_count] == _ids(previous)[-segment.overlap_count :]
            assert segment.overlap_count == 3
    assert primary == [f"m{index}" for index in range(61)]
    assert closed[0].opened_by == "start"
    assert closed[0].closed_by == "max_messages"
    assert closed[-1].closed_by == "end"


def test_time_gap_boundary_and_null_timestamps():
    params = SegmentParams(gap_seconds=1800, max_messages=50, overlap_messages=4)
    exact = _close_all(
        _refs(
            [
                {"message_id": "a", "timestamp": 0},
                {"message_id": "b", "timestamp": 10},
                {"message_id": "c", "timestamp": 20},
                {"message_id": "d", "timestamp": 20 + 1800},
                {"message_id": "e", "timestamp": 20 + 1800 + 5},
            ]
        ),
        params,
    )
    assert [_ids(segment) for segment in exact] == [["a", "b", "c"], ["d", "e"]]
    assert exact[1].opened_by == "time_gap"
    assert exact[1].overlap_count == 0

    almost = _close_all(
        _refs(
            [
                {"message_id": "a", "timestamp": 0},
                {"message_id": "b", "timestamp": 1799},
            ]
        ),
        params,
    )
    assert len(almost) == 1

    missing_time = _close_all(
        _refs(
            [
                {"message_id": "a", "timestamp": 0},
                {"message_id": "b", "timestamp": None},
                {"message_id": "c", "timestamp": 10_000},
            ]
        ),
        params,
    )
    assert len(missing_time) == 1


def test_overlap_is_not_carried_across_conversations_or_gaps_unless_asked():
    params = SegmentParams(gap_seconds=1800, max_messages=10, overlap_messages=3)
    messages = _refs(
        [{"message_id": f"a{index}", "conversation_id": "A", "timestamp": index} for index in range(12)]
        + [{"message_id": f"b{index}", "conversation_id": "B", "timestamp": index} for index in range(2)]
    )
    closed = _close_all(messages, params)
    b_segments = [segment for segment in closed if segment.conversation_id == "B"]
    assert len(b_segments) == 1
    assert b_segments[0].opened_by == "conversation_change"
    assert _ids(b_segments[0]) == ["b0", "b1"]
    a_ids = {message_id for segment in closed if segment.conversation_id == "A" for message_id in _ids(segment)}
    assert a_ids.isdisjoint(set(_ids(b_segments[0])))

    with_gap_overlap = SegmentParams(
        gap_seconds=1800,
        max_messages=50,
        overlap_messages=2,
        overlap_on_time_gap=True,
    )
    gapped = _close_all(
        _refs(
            [
                {"message_id": "a", "timestamp": 1},
                {"message_id": "b", "timestamp": 2},
                {"message_id": "c", "timestamp": 3},
                {"message_id": "d", "timestamp": 3 + 1800},
            ]
        ),
        with_gap_overlap,
    )
    assert _ids(gapped[1]) == ["b", "c", "d"]
    assert gapped[1].overlap_count == 2
    assert [item.message_id for item in gapped[1].primary_messages] == ["d"]


def test_sqlite_segments_match_the_splitter_and_page_size(tmp_path: Path):
    items = [{"message_id": f"m{index}", "timestamp": index, "content": f"text {index}"} for index in range(61)]
    params = SegmentParams(gap_seconds=1800, max_messages=10, overlap_messages=3)
    expected = [_ids(segment) for segment in _close_all(_refs(items), params)]

    wide = tmp_path / "wide.db"
    narrow = tmp_path / "narrow.db"
    seed_messages(wide, items)
    seed_messages(narrow, items)
    wide_result = run_segmentation(wide, params=params, page_size=1000)
    narrow_result = run_segmentation(narrow, params=params, page_size=4)
    assert wide_result["status"] == "completed"
    assert narrow_result["status"] == "completed"
    assert _stored_ids(wide) == expected
    assert _stored_ids(narrow) == expected
    _assert_segment_rows(wide, [item["message_id"] for item in items], max_messages=10)


def test_resume_matches_a_full_run(tmp_path: Path):
    items = [{"message_id": f"m{index}", "timestamp": index, "content": f"text {index}"} for index in range(61)]
    params = SegmentParams(gap_seconds=1800, max_messages=10, overlap_messages=3)
    full = tmp_path / "full.db"
    paused = tmp_path / "paused.db"
    seed_messages(full, items)
    seed_messages(paused, items)
    run_segmentation(full, params=params, page_size=1000)
    first = run_segmentation(paused, params=params, page_size=5, max_segments=2)
    assert first["status"] == "running"
    assert first["segments_written"] == 2
    assert first["segment_count"] == 2
    second = run_segmentation(paused, params=params, page_size=3)
    assert second["status"] == "completed"
    assert second["resumed"] is True
    assert second["messages_consumed"] == 61
    assert _stored_ids(paused) == _stored_ids(full)

    again = run_segmentation(paused, params=params)
    assert again["unchanged"] is True
    assert again["segments_written"] == 0
    assert again["segment_count"] == second["segment_count"]


def test_checkpoint_refuses_a_changed_message_table_until_rebuild(tmp_path: Path):
    db_path = tmp_path / "grow.db"
    seed_messages(db_path, [{"message_id": "m0", "timestamp": 1, "content": "one"}])
    params = SegmentParams(gap_seconds=1800, max_messages=10, overlap_messages=2)
    run_segmentation(db_path, params=params)
    insert_message(
        db_path,
        {"message_id": "m1", "timestamp": 2, "content": "two", "source_position": 1},
    )
    with pytest.raises(SegmentationConflictError):
        run_segmentation(db_path, params=params)
    with pytest.raises(SegmentationConflictError):
        run_segmentation(db_path, params=SegmentParams(gap_seconds=60, max_messages=10, overlap_messages=2))
    rebuilt = run_segmentation(db_path, params=params, rebuild=True)
    assert rebuilt["message_total"] == 2
    assert rebuilt["segment_count"] == 1
    conn = init_db(db_path)
    try:
        ids = json.loads(conn.execute("SELECT message_ids_json FROM conversation_segments").fetchone()["message_ids_json"])
    finally:
        conn.close()
    assert ids == ["m0", "m1"]


def test_fixture_segments_reference_imported_message_ids(tmp_path: Path):
    db_path = tmp_path / "sample.db"
    run_import(FIXTURE, db_path)
    result = run_segmentation(db_path)
    assert result["status"] == "completed"
    assert result["message_total"] == 300
    assert result["segment_count"] >= 2
    conn = init_db(db_path)
    try:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(conversation_segments)")}
        assert "message_ids_json" in columns
        assert "target_message_count" in columns
        assert "processing_status" in columns
        assert "content" not in columns
        message_ids = [
            row["message_id"]
            for row in conn.execute("SELECT message_id FROM messages ORDER BY source_position")
        ]
        _assert_segment_rows(db_path, message_ids, conn=conn)
        gap_rows = conn.execute(
            """
            SELECT opened_by, overlap_message_ids_json, start_time
            FROM conversation_segments
            WHERE opened_by = 'time_gap'
            """
        ).fetchall()
        for row in gap_rows:
            assert json.loads(row["overlap_message_ids_json"]) == []
            assert row["start_time"] is not None
    finally:
        conn.close()


def test_segment_and_status_cli(tmp_path: Path):
    db_path = tmp_path / "cli.db"
    run_import(FIXTURE, db_path)
    segmented = subprocess.run(
        [sys.executable, "-m", "scripts.segment_conversations", "--db", str(db_path), "--page-size", "20"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert segmented.returncode == 0, segmented.stderr
    assert "status: completed" in segmented.stdout
    assert "segment_count:" in segmented.stdout
    status = subprocess.run(
        [sys.executable, "-m", "scripts.show_status", "--db", str(db_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert status.returncode == 0, status.stderr
    assert "segments:" in status.stdout
    assert "pending:" in status.stdout
    assert "segmentation: completed" in status.stdout
    missing = subprocess.run(
        [sys.executable, "-m", "scripts.segment_conversations", "--db", str(tmp_path / "missing.db")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 1


def _stored_ids(db_path: Path) -> list[list[str]]:
    conn = init_db(db_path)
    try:
        rows = conn.execute(
            """
            SELECT s.message_ids_json
            FROM conversation_segments s
            JOIN conversation_segment_messages sm
              ON sm.segment_id = s.segment_id AND sm.position = 0
            JOIN messages m ON m.message_id = sm.message_id
            ORDER BY COALESCE(m.conversation_id, ''), m.source_position, s.segment_id
            """
        ).fetchall()
        return [json.loads(row["message_ids_json"]) for row in rows]
    finally:
        conn.close()


def _assert_segment_rows(
    db_path: Path,
    expected_ids: list[str],
    conn: sqlite3.Connection | None = None,
    *,
    max_messages: int = 50,
) -> None:
    own = conn is None
    if conn is None:
        conn = init_db(db_path)
    try:
        rows = conn.execute(
            """
            SELECT s.*, m.source_position
            FROM conversation_segments s
            JOIN conversation_segment_messages sm
              ON sm.segment_id = s.segment_id AND sm.position = 0
            JOIN messages m ON m.message_id = sm.message_id
            ORDER BY COALESCE(m.conversation_id, ''), m.source_position, s.segment_id
            """
        ).fetchall()
        assert rows
        primary: list[str] = []
        previous_ids: list[str] | None = None
        for row in rows:
            message_ids = json.loads(row["message_ids_json"])
            overlap_ids = json.loads(row["overlap_message_ids_json"])
            assert isinstance(message_ids, list)
            assert message_ids
            assert all(isinstance(message_id, str) for message_id in message_ids)
            assert message_ids[0] == row["start_message_id"]
            assert message_ids[-1] == row["end_message_id"]
            assert row["message_count"] == len(message_ids)
            assert message_ids[: len(overlap_ids)] == overlap_ids
            assert row["target_message_count"] == len(message_ids) - len(overlap_ids)
            assert row["message_count"] <= max_messages
            assert row["processing_status"] == "pending"
            linked = conn.execute(
                """
                SELECT message_id, is_overlap
                FROM conversation_segment_messages
                WHERE segment_id = ?
                ORDER BY position
                """,
                (row["segment_id"],),
            ).fetchall()
            assert [item["message_id"] for item in linked] == message_ids
            assert [item["message_id"] for item in linked if item["is_overlap"]] == overlap_ids
            found = conn.execute(
                f"SELECT COUNT(*) AS n FROM messages WHERE message_id IN ({','.join('?' for _ in message_ids)})",
                message_ids,
            ).fetchone()["n"]
            assert found == len(message_ids)
            if overlap_ids:
                assert previous_ids is not None
                assert overlap_ids == previous_ids[-len(overlap_ids) :]
            primary.extend(message_ids[len(overlap_ids) :])
            previous_ids = message_ids
        assert primary == expected_ids
        target_sum = conn.execute("SELECT SUM(target_message_count) AS n FROM conversation_segments").fetchone()["n"]
        assert target_sum == len(expected_ids)
    finally:
        if own:
            conn.close()
