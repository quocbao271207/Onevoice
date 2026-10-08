from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import audit_datasets, materialize_audio, sanitize_asr_report_urls
from scripts.merge_manifests import merge_task


def asr_spec() -> dict:
    return {
        "repo_id": "owner/dataset",
        "config": "default",
        "task": "asr",
        "language": "vi",
        "domain": "medical",
        "license": "mit",
        "train_splits": ["train"],
        "validation_splits": [],
        "test_splits": [],
        "calibration_splits": [],
        "text_column": "text",
        "audio_column": "audio",
        "id_column": "native_id",
        "max_exact_duplicate_rate": 1.0,
    }


def source_row() -> dict:
    return {
        "native_id": "native-1",
        "text": "Bệnh nhân đau ngực",
        "audio": {"url": public_signed_url()},
    }


def public_signed_url() -> str:
    return "https://datasets-server.huggingface.co/audio?X-Amz-Signature=" + "A" * 64


def test_audit_never_persists_signed_audio_urls(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        audit_datasets,
        "split_sizes",
        lambda _repo: [{"config": "default", "split": "train", "num_rows": 1}],
    )
    monkeypatch.setattr(
        audit_datasets,
        "iter_rows",
        lambda *_args, **_kwargs: iter([source_row()]),
    )

    audit_datasets.audit_asr("sample", asr_spec(), tmp_path, max_rows=None)

    record = json.loads((tmp_path / "records" / "sample--train.jsonl").read_text(encoding="utf-8"))
    listening = json.loads((tmp_path / "listening" / "sample.jsonl").read_text(encoding="utf-8"))
    assert "audio_url" not in record
    assert "audio_url" not in listening


def test_refresh_uses_url_only_in_memory_and_download_result_strips_it(
    tmp_path: Path,
    monkeypatch,
):
    spec = asr_spec()
    durable = audit_datasets.durable_asr_record(
        audit_datasets.normalize_asr_row("sample", spec, "train", source_row())
    )
    monkeypatch.setattr(
        materialize_audio,
        "split_sizes",
        lambda _repo: [{"config": "default", "split": "train", "num_rows": 1}],
    )
    monkeypatch.setattr(
        materialize_audio,
        "api_json",
        lambda _endpoint, _params: {"rows": [{"row": source_row()}]},
    )

    refreshed = materialize_audio.refresh_audio_urls([durable], {"sample": spec})
    assert "audio_url" in refreshed[0]
    destination = materialize_audio.destination_path(refreshed[0], tmp_path / "audio")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"W" * 45)

    completed = materialize_audio.download_one(refreshed[0], tmp_path / "audio")
    assert "audio_url" not in completed
    assert completed["audio_path"] == str(destination.resolve())


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "http://dataset.example/audio.wav",
        "https://user:password@dataset.example/audio.wav",
        "https://localhost/audio.wav",
        "https://127.0.0.1/audio.wav",
        "https://example.com/audio.wav",
        "https://datasets-server.huggingface.co:444/audio.wav",
    ],
)
def test_download_rejects_unsafe_url_before_network(
    tmp_path: Path,
    monkeypatch,
    unsafe_url: str,
):
    called = False

    def unexpected_network(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("network must not be called")

    monkeypatch.setattr(materialize_audio, "urlopen", unexpected_network)
    record = {
        "id": "record-1",
        "source": "sample",
        "source_split": "train",
        "audio_url": unsafe_url,
    }
    with pytest.raises(ValueError, match="Audio download URL") as caught:
        materialize_audio.download_one(record, tmp_path)
    assert unsafe_url not in str(caught.value)
    assert not called


def test_download_failure_does_not_echo_url_and_removes_partial_file(
    tmp_path: Path,
    monkeypatch,
):
    signed_url = public_signed_url()

    def fail_without_network(_request, timeout):
        assert timeout == 120
        raise OSError(signed_url)

    monkeypatch.setattr(materialize_audio, "urlopen", fail_without_network)
    monkeypatch.setattr(materialize_audio.time, "sleep", lambda _seconds: None)
    record = {
        "id": "record-1",
        "source": "sample",
        "source_split": "train",
        "audio_url": signed_url,
    }
    destination = materialize_audio.destination_path(record, tmp_path)
    destination.parent.mkdir(parents=True)
    destination.with_suffix(".part").write_bytes(b"partial")

    with pytest.raises(RuntimeError, match=r"1 attempts \(OSError\)") as caught:
        materialize_audio.download_one(record, tmp_path, retries=1)
    assert signed_url not in str(caught.value)
    assert not destination.with_suffix(".part").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", "../escape"),
        ("source_split", "a/b"),
        ("id", ".."),
    ],
)
def test_audio_destination_rejects_path_traversal(field: str, value: str, tmp_path: Path):
    record = {
        "id": "record-1",
        "source": "sample",
        "source_split": "train",
    }
    record[field] = value
    with pytest.raises(ValueError, match="path component"):
        materialize_audio.destination_path(record, tmp_path)


def test_download_rejects_linked_partial_file(tmp_path: Path):
    if not hasattr(Path, "symlink_to"):
        pytest.skip("symlink API unavailable")
    record = {
        "id": "record-1",
        "source": "sample",
        "source_split": "train",
        "audio_url": public_signed_url(),
    }
    destination = materialize_audio.destination_path(record, tmp_path / "audio")
    destination.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"protected")
    partial = destination.with_suffix(".part")
    try:
        partial.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation unavailable")

    with pytest.raises(ValueError, match="linked audio path"):
        materialize_audio.download_one(record, tmp_path / "audio")
    assert outside.read_bytes() == b"protected"


def test_download_enforces_stream_size_limit_and_removes_partial_file(
    tmp_path: Path,
    monkeypatch,
):
    class Response:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def geturl(self):
            return public_signed_url()

        def read(self, _size):
            if getattr(self, "sent", False):
                return b""
            self.sent = True
            return b"A" * 9

    monkeypatch.setattr(materialize_audio, "MAX_AUDIO_DOWNLOAD_BYTES", 8)
    monkeypatch.setattr(materialize_audio, "urlopen", lambda *_args, **_kwargs: Response())
    record = {
        "id": "record-1",
        "source": "sample",
        "source_split": "train",
        "audio_url": public_signed_url(),
    }
    destination = materialize_audio.destination_path(record, tmp_path)

    with pytest.raises(RuntimeError, match=r"1 attempts \(OSError\)"):
        materialize_audio.download_one(record, tmp_path, retries=1)
    assert not destination.with_suffix(".part").exists()


def test_merge_strips_legacy_audio_urls(tmp_path: Path):
    records_dir = tmp_path / "records"
    output_dir = tmp_path / "merged"
    records_dir.mkdir()
    record = audit_datasets.normalize_asr_row("sample", asr_spec(), "train", source_row())
    (records_dir / "sample--train.jsonl").write_text(
        json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result = merge_task(
        "asr",
        [("sample", asr_spec())],
        records_dir,
        output_dir,
    )

    merged = json.loads((output_dir / "asr--train.jsonl").read_text(encoding="utf-8"))
    assert result["kept"] == {"train": 1}
    assert "audio_url" not in merged


def test_report_sanitizer_is_dry_run_by_default_and_atomic_on_apply(tmp_path: Path):
    report = tmp_path / "records.jsonl"
    payload = (
        json.dumps({"id": "one", "audio_url": public_signed_url(), "text": "xin chào"}, ensure_ascii=False)
        + "\n"
        + json.dumps({"id": "two", "text": "đau ngực"}, ensure_ascii=False)
        + "\n"
    )
    report.write_text(payload, encoding="utf-8")

    assert sanitize_asr_report_urls.sanitize_report(report, apply=False) == 1
    assert report.read_text(encoding="utf-8") == payload
    assert not report.with_name(f".{report.name}.sanitize.part").exists()

    assert sanitize_asr_report_urls.sanitize_report(report, apply=True) == 1
    rows = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"id": "one", "text": "xin chào"},
        {"id": "two", "text": "đau ngực"},
    ]
    assert sanitize_asr_report_urls.sanitize_report(report, apply=False) == 0


def test_report_sanitizer_rejects_malformed_or_oversized_records(tmp_path: Path):
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_bytes(b"not-json\n")
    with pytest.raises(ValueError, match="Invalid UTF-8 JSONL"):
        sanitize_asr_report_urls.sanitize_report(malformed, apply=False)

    oversized = tmp_path / "oversized.jsonl"
    oversized.write_bytes(b'{' + b'"text":"' + b"A" * sanitize_asr_report_urls.MAX_JSONL_LINE_BYTES)
    with pytest.raises(ValueError, match="exceeds size limit"):
        sanitize_asr_report_urls.sanitize_report(oversized, apply=False)
