from __future__ import annotations

import base64
import http.client
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import demo.demo_web as demo_web


def mt_result(*, safe: bool, translated: str = "safe") -> SimpleNamespace:
    return SimpleNamespace(
        translated_text=translated,
        target_lang="en",
        safety_passed=safe,
        safety_issues=[] if safe else ["negation_mismatch:private-detail"],
        requires_confirmation=False,
        from_cache=False,
        latency_ms=12.5,
    )


def speech_result(
    *,
    safe: bool = True,
    translated: str = "Check pulse",
    confirmation: bool = False,
    output_audio=None,
    degraded_mode=None,
) -> SimpleNamespace:
    return SimpleNamespace(
        asr_text="Kiểm tra mạch",
        asr_language="vi",
        translated_text=translated,
        target_language="en",
        safety_passed=safe,
        safety_issues=[] if safe else ["dose_mismatch:private-detail"],
        requires_confirmation=confirmation,
        from_cache=confirmation,
        output_audio=output_audio,
        degraded_mode=degraded_mode,
        degradation_code="tts_runtime_error" if degraded_mode else None,
        total_latency_ms=20.0,
    )


def test_web_text_payload_hides_unsafe_candidate():
    class Pipeline:
        def translate_text(self, text, language):
            assert (text, language) == ("Không dùng thuốc", "vi")
            return mt_result(safe=False, translated="PATIENT_SECRET_UNSAFE")

    payload = demo_web.translate_text_payload(
        Pipeline(),
        {"text": "Không dùng thuốc", "source_lang": "vi"},
    )

    assert payload["safety_passed"] is False
    assert payload["translated_text"] == ""
    assert payload["safety_issue_codes"] == ["negation_mismatch"]
    assert "PATIENT_SECRET_UNSAFE" not in json.dumps(payload)


def test_web_audio_is_bounded_cleaned_and_never_exposes_unsafe_translation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    observed_paths = []

    def fake_load(path, *, max_duration_seconds):
        assert max_duration_seconds == 30.0
        assert path.is_file()
        observed_paths.append(path)
        return np.zeros(160, dtype=np.float32), 16_000

    class Pipeline:
        asr_engine = SimpleNamespace(max_input_duration_seconds=30.0)

        def translate_speech(self, audio, *, source_lang, sample_rate):
            assert audio.shape == (160,)
            assert source_lang is None
            assert sample_rate == 16_000
            return speech_result(safe=False, translated="PATIENT_SECRET_UNSAFE")

    monkeypatch.setattr(demo_web, "load_audio_file", fake_load)
    store = demo_web.PendingPlaybackStore()
    payload = demo_web.translate_audio_payload(
        Pipeline(),
        store,
        {
            "audio_base64": base64.b64encode(b"audio").decode("ascii"),
            "source_lang": None,
        },
    )

    assert observed_paths and not observed_paths[0].exists()
    assert payload["translated_text"] == ""
    assert payload["playback_available"] is False
    assert payload["request_id"] is None
    assert "PATIENT_SECRET_UNSAFE" not in json.dumps(payload)

    with pytest.raises(ValueError, match="invalid"):
        demo_web.translate_audio_payload(
            Pipeline(),
            store,
            {"audio_base64": "not base64!", "source_lang": "vi"},
        )


def test_web_playback_tokens_are_one_time_expiring_and_policy_bound():
    now = [100.0]
    store = demo_web.PendingPlaybackStore(
        capacity=1,
        ttl_seconds=5.0,
        clock=lambda: now[0],
    )
    regular = speech_result(output_audio=np.zeros(16, dtype=np.float32))
    token = store.put(regular)
    calls = []

    class Pipeline:
        def play(self, result):
            calls.append(("play", result))

        def confirm_and_play(self, result, *, confirmed):
            calls.append(("confirm", result, confirmed))

    assert demo_web.perform_playback(
        Pipeline(), store, {"request_id": token, "confirmed": False}
    ) == {"played": True}
    assert calls == [("play", regular)]
    with pytest.raises(ValueError, match="expired or unknown"):
        store.take(token, confirmed=False)

    clinical = speech_result(confirmation=True)
    clinical_token = store.put(clinical)
    with pytest.raises(ValueError, match="does not match"):
        store.take(clinical_token, confirmed=False)
    assert store.take(clinical_token, confirmed=True) is clinical

    expired = store.put(regular)
    now[0] = 106.0
    with pytest.raises(ValueError, match="expired or unknown"):
        store.take(expired, confirmed=False)


def test_web_server_is_loopback_csrf_protected_and_no_store():
    csrf = "c" * 32

    class Pipeline:
        def get_status(self):
            return {
                "ready": True,
                "operational_mode": "text_only",
                "degradation_code": "tts_runtime_error",
                "components": {"mt": True},
            }

        def translate_text(self, text, language):
            assert (text, language) == ("Xin chào", "vi")
            return mt_result(safe=True, translated="Hello")

    server = demo_web.create_server(Pipeline(), port=0, csrf_token=csrf)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        connection = http.client.HTTPConnection(host, port, timeout=3)
        connection.request("GET", "/")
        response = connection.getresponse()
        page = response.read().decode("utf-8")
        assert response.status == 200
        assert response.getheader("Cache-Control") == "no-store"
        assert response.getheader("X-Frame-Options") == "DENY"
        assert "Python" not in (response.getheader("Server") or "")
        assert "Content-Security-Policy" in response.headers
        assert csrf in page
        connection.close()

        connection = http.client.HTTPConnection(host, port, timeout=3)
        connection.request("GET", "/", headers={"Host": "attacker.invalid"})
        response = connection.getresponse()
        assert response.status == 421
        assert json.loads(response.read())["error"] == "invalid_host"
        connection.close()

        body = json.dumps({"text": "Xin chào", "source_lang": "vi"})
        connection = http.client.HTTPConnection(host, port, timeout=3)
        connection.request(
            "POST",
            "/api/translate-text",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert response.status == 403
        assert json.loads(response.read())["error"] == "forbidden"
        connection.close()

        connection = http.client.HTTPConnection(host, port, timeout=3)
        connection.request(
            "POST",
            "/api/translate-text",
            body=body,
            headers={
                "Content-Type": "application/json",
                "X-OneVoice-CSRF": csrf,
            },
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert payload["translated_text"] == "Hello"
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert not thread.is_alive()
