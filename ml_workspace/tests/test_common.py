import json

import pytest

from common import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    build_metrics_payload,
    find_jsonl_files,
    load_jsonl_dataset,
    parse_record,
    write_json,
)


def test_label_mapping_is_stable():
    assert LABELS == ("negative", "neutral", "positive")
    assert LABEL2ID == {"negative": 0, "neutral": 1, "positive": 2}
    assert ID2LABEL == {0: "negative", 1: "neutral", 2: "positive"}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ({"text": "ok", "label": "positive"}, ("ok", "positive")),
        ({"text": "ok", "label": "POSITIVE"}, ("ok", "positive")),
        ({"text": "ok", "label": " neutral "}, ("ok", "neutral")),
        ({"text": "ok", "label": "positive", "_extra": 1}, ("ok", "positive")),
    ],
)
def test_parse_record_valid(raw, expected):
    assert parse_record(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        {"text": "", "label": "positive"},
        {"text": "   ", "label": "positive"},
        {"text": None, "label": "positive"},
        {"text": "ok", "label": None},
        {"text": "ok", "label": "happy"},
        {"text": "ok"},
        {"label": "positive"},
    ],
)
def test_parse_record_invalid(raw):
    assert parse_record(raw) is None


def test_load_jsonl_dataset_directory(tmp_path):
    f1 = tmp_path / "a.jsonl"
    f1.write_text(
        "\n".join(
            [
                json.dumps({"text": "good", "label": "positive"}),
                json.dumps({"text": "bad", "label": "negative"}),
                "",  # riga vuota
                "{not-json",  # malformato
                json.dumps({"text": "ok", "label": "happy"}),  # label invalida
            ]
        ),
        encoding="utf-8",
    )
    f2 = tmp_path / "b.jsonl"
    f2.write_text(json.dumps({"text": "meh", "label": "neutral"}) + "\n", encoding="utf-8")

    records, report = load_jsonl_dataset(tmp_path)

    assert len(records) == 3
    assert {r["label"] for r in records} == {"positive", "negative", "neutral"}
    assert all(r["label_id"] == LABEL2ID[r["label"]] for r in records)
    assert report.read == 5
    assert report.valid == 3
    assert report.skipped == 2
    assert report.skipped_reasons == {
        "invalid_json": 1,
        "invalid_text_or_label": 1,
    }
    assert sorted(report.files) == [str(f1), str(f2)]


def test_load_jsonl_dataset_single_file(tmp_path):
    f = tmp_path / "single.jsonl"
    f.write_text(json.dumps({"text": "x", "label": "positive"}) + "\n", encoding="utf-8")
    records, report = load_jsonl_dataset(f)
    assert len(records) == 1
    assert report.files == [str(f)]


def test_find_jsonl_files_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_jsonl_files(tmp_path / "missing")


def test_find_jsonl_files_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_jsonl_files(tmp_path)


def test_build_metrics_payload_complete():
    payload = build_metrics_payload(
        model_version="v1.1.0-rc1",
        eval_dataset="test.parquet",
        accuracy=0.85,
        f1_macro=0.83,
        f1_per_label={"negative": 0.8, "neutral": 0.82, "positive": 0.87},
        eval_loss=0.42,
        num_samples=1000,
    )
    assert payload == {
        "model_version": "v1.1.0-rc1",
        "eval_dataset": "test.parquet",
        "accuracy": 0.85,
        "f1_macro": 0.83,
        "f1_negative": 0.8,
        "f1_neutral": 0.82,
        "f1_positive": 0.87,
        "eval_loss": 0.42,
        "num_samples": 1000,
    }
    # Tutti i tipi devono essere coerenti con ModelMetricsPayload
    for k in ("accuracy", "f1_macro", "f1_negative", "f1_neutral", "f1_positive", "eval_loss"):
        assert isinstance(payload[k], float)
    assert isinstance(payload["num_samples"], int)


def test_build_metrics_payload_missing_label_raises():
    with pytest.raises(ValueError, match="f1_per_label"):
        build_metrics_payload(
            model_version="v1",
            eval_dataset="d",
            accuracy=0.5,
            f1_macro=0.5,
            f1_per_label={"negative": 0.5, "neutral": 0.5},  # manca positive
            eval_loss=0.1,
            num_samples=10,
        )


def test_write_json_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "deep" / "report.json"
    write_json(target, {"a": 1, "b": [1, 2, 3]})
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": [1, 2, 3]}


def test_post_metrics_to_api_handles_connection_error(monkeypatch):
    """Se l'API è down, post_metrics_to_api non solleva ma ritorna sent=False."""
    import httpx

    from evaluate import post_metrics_to_api

    def raise_connect(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", raise_connect)

    result = post_metrics_to_api("http://localhost:9999", {"x": 1})
    assert result["sent"] is False
    assert "connection refused" in result["error"]
    assert result["url"] == "http://localhost:9999/model/metrics"


def test_post_metrics_to_api_success(monkeypatch):
    import httpx

    from evaluate import post_metrics_to_api

    class FakeResp:
        status_code = 201

        def raise_for_status(self):
            return None

    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResp()

    monkeypatch.setattr(httpx, "post", fake_post)

    payload = {"model_version": "v1", "accuracy": 0.9}
    result = post_metrics_to_api("http://api:8000", payload)
    assert result == {
        "sent": True,
        "url": "http://api:8000/model/metrics",
        "status_code": 201,
    }
    assert captured["url"] == "http://api:8000/model/metrics"
    assert captured["json"] == payload
