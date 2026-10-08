"""Loopback-only web demo for the verified OneVoice pipeline.

The browser is a presentation surface only. Clinical safety and playback
confirmation remain server-side invariants owned by ``MediVoicePipeline``.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import json
import logging
import math
import os
import secrets
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from numbers import Real
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from demo.demo_cli import load_audio_file  # noqa: E402
from src.pipeline.safety_guard import safety_issue_codes  # noqa: E402


logger = logging.getLogger("OneVoiceWeb")

LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_WEB_AUDIO_BYTES = 16 * 1024 * 1024
MAX_WEB_REQUEST_BYTES = 23 * 1024 * 1024
MAX_PENDING_PLAYBACKS = 8
PLAYBACK_TTL_SECONDS = 120.0
SUPPORTED_LANGUAGES = {"vi", "en"}


class PendingPlaybackStore:
    """Bounded, expiring, one-time storage for safety-checked results."""

    def __init__(
        self,
        *,
        capacity: int = MAX_PENDING_PLAYBACKS,
        ttl_seconds: float = PLAYBACK_TTL_SECONDS,
        clock=time.monotonic,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Playback capacity must be a positive integer")
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, Real)
            or not math.isfinite(float(ttl_seconds))
            or float(ttl_seconds) <= 0.0
        ):
            raise ValueError("Playback TTL must be finite and positive")
        self._capacity = capacity
        self._ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._items: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def _purge_locked(self, now: float) -> None:
        expired = [token for token, (deadline, _) in self._items.items() if deadline <= now]
        for token in expired:
            self._items.pop(token, None)

    def put(self, result: Any) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            now = float(self._clock())
            self._purge_locked(now)
            while len(self._items) >= self._capacity:
                self._items.popitem(last=False)
            self._items[token] = (now + self._ttl_seconds, result)
        return token

    def take(self, token: object, *, confirmed: object) -> Any:
        if not isinstance(token, str) or not token or len(token) > 256:
            raise ValueError("Playback request token is invalid")
        if type(confirmed) is not bool:
            raise ValueError("Playback confirmation must be boolean")
        with self._lock:
            now = float(self._clock())
            self._purge_locked(now)
            record = self._items.get(token)
            if record is None:
                raise ValueError("Playback request is expired or unknown")
            result = record[1]
            requires_confirmation = getattr(result, "requires_confirmation", None)
            if type(requires_confirmation) is not bool:
                raise RuntimeError("Playback result confirmation state is invalid")
            if confirmed is not requires_confirmation:
                raise ValueError("Playback confirmation does not match result policy")
            self._items.pop(token)
            return result


def _finite_non_negative(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise RuntimeError(f"Pipeline returned invalid {label}")
    return float(value)


def _strict_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise RuntimeError(f"Pipeline returned invalid {label}")
    return value


def _validate_language(value: object, *, allow_auto: bool) -> str | None:
    if allow_auto and value is None:
        return None
    if not isinstance(value, str) or value not in SUPPORTED_LANGUAGES:
        raise ValueError("source_lang must be 'vi', 'en', or null for audio")
    return value


def _safety_payload(result: Any) -> tuple[bool, list[str]]:
    passed = _strict_bool(getattr(result, "safety_passed", None), "safety state")
    issues = getattr(result, "safety_issues", None)
    if not isinstance(issues, list) or any(not isinstance(item, str) for item in issues):
        raise RuntimeError("Pipeline returned invalid safety evidence")
    codes = list(safety_issue_codes(issues))
    if (passed and codes) or (not passed and not codes):
        raise RuntimeError("Pipeline returned inconsistent safety evidence")
    return passed, codes


def translate_text_payload(pipeline: Any, body: object) -> dict[str, object]:
    if not isinstance(body, dict) or set(body) != {"text", "source_lang"}:
        raise ValueError("Text request schema is invalid")
    text = body.get("text")
    source_lang = _validate_language(body.get("source_lang"), allow_auto=False)
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    result = pipeline.translate_text(text, source_lang)
    safety_passed, issue_codes = _safety_payload(result)
    translated = getattr(result, "translated_text", None)
    if not isinstance(translated, str):
        raise RuntimeError("Pipeline returned invalid translated text")
    target_lang = getattr(result, "target_lang", None)
    if target_lang not in SUPPORTED_LANGUAGES:
        raise RuntimeError("Pipeline returned invalid target language")
    if target_lang == source_lang or (safety_passed and not translated.strip()):
        raise RuntimeError("Pipeline returned inconsistent text translation")
    requires_confirmation = _strict_bool(
        getattr(result, "requires_confirmation", None),
        "confirmation state",
    )
    from_cache = _strict_bool(getattr(result, "from_cache", None), "cache state")
    return {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "translated_text": translated if safety_passed else "",
        "safety_passed": safety_passed,
        "safety_issue_codes": issue_codes,
        "requires_confirmation": requires_confirmation,
        "from_cache": from_cache,
        "latency_ms": _finite_non_negative(getattr(result, "latency_ms", None), "latency"),
    }


def _decode_audio(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("audio_base64 must be a non-empty string")
    maximum_encoded = ((MAX_WEB_AUDIO_BYTES + 2) // 3) * 4
    if len(value) > maximum_encoded:
        raise ValueError("Encoded audio exceeds the web demo limit")
    try:
        payload = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("audio_base64 is invalid") from error
    if not payload or len(payload) > MAX_WEB_AUDIO_BYTES:
        raise ValueError("Decoded audio exceeds the web demo limit")
    return payload


def translate_audio_payload(
    pipeline: Any,
    playback_store: PendingPlaybackStore,
    body: object,
) -> dict[str, object]:
    if not isinstance(body, dict) or set(body) != {"audio_base64", "source_lang"}:
        raise ValueError("Audio request schema is invalid")
    source_lang = _validate_language(body.get("source_lang"), allow_auto=True)
    audio_bytes = _decode_audio(body.get("audio_base64"))
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as temporary:
            temporary.write(audio_bytes)
            temporary_path = Path(temporary.name)
        if os.name == "posix":
            temporary_path.chmod(0o600)
        audio, sample_rate = load_audio_file(
            temporary_path,
            max_duration_seconds=pipeline.asr_engine.max_input_duration_seconds,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    result = pipeline.translate_speech(
        audio,
        source_lang=source_lang,
        sample_rate=sample_rate,
    )
    safety_passed, issue_codes = _safety_payload(result)
    translated = getattr(result, "translated_text", None)
    asr_text = getattr(result, "asr_text", None)
    asr_language = getattr(result, "asr_language", None)
    target_language = getattr(result, "target_language", None)
    if not isinstance(asr_text, str) or not isinstance(translated, str):
        raise RuntimeError("Pipeline returned invalid speech text")
    if asr_language not in SUPPORTED_LANGUAGES or target_language not in SUPPORTED_LANGUAGES:
        raise RuntimeError("Pipeline returned invalid speech language")
    requires_confirmation = _strict_bool(
        getattr(result, "requires_confirmation", None),
        "confirmation state",
    )
    from_cache = _strict_bool(getattr(result, "from_cache", None), "cache state")
    degraded_mode = getattr(result, "degraded_mode", None)
    degradation_code = getattr(result, "degradation_code", None)
    if degraded_mode not in {None, "text_only"}:
        raise RuntimeError("Pipeline returned invalid degradation mode")
    if degradation_code is not None and (
        not isinstance(degradation_code, str) or len(degradation_code) > 128
    ):
        raise RuntimeError("Pipeline returned invalid degradation code")
    if (degraded_mode == "text_only") is not isinstance(degradation_code, str):
        raise RuntimeError("Pipeline returned inconsistent degradation evidence")
    if target_language == asr_language or (safety_passed and not translated.strip()):
        raise RuntimeError("Pipeline returned inconsistent speech translation")
    output_audio = getattr(result, "output_audio", None)
    if requires_confirmation and output_audio is not None:
        raise RuntimeError("Confirmation-gated result must not contain audio")
    playback_available = (
        safety_passed
        and degraded_mode != "text_only"
        and (output_audio is not None or requires_confirmation)
    )
    request_id = playback_store.put(result) if playback_available else None
    return {
        "request_id": request_id,
        "asr_text": asr_text,
        "source_lang": asr_language,
        "target_lang": target_language,
        "translated_text": translated if safety_passed else "",
        "safety_passed": safety_passed,
        "safety_issue_codes": issue_codes,
        "requires_confirmation": requires_confirmation,
        "playback_available": playback_available,
        "from_cache": from_cache,
        "degraded_mode": degraded_mode,
        "degradation_code": degradation_code,
        "latency_ms": _finite_non_negative(
            getattr(result, "total_latency_ms", None),
            "total latency",
        ),
    }


def perform_playback(
    pipeline: Any,
    playback_store: PendingPlaybackStore,
    body: object,
) -> dict[str, object]:
    if not isinstance(body, dict) or set(body) != {"request_id", "confirmed"}:
        raise ValueError("Playback request schema is invalid")
    confirmed = body.get("confirmed")
    result = playback_store.take(body.get("request_id"), confirmed=confirmed)
    if confirmed is True:
        pipeline.confirm_and_play(result, confirmed=True)
    else:
        pipeline.play(result)
    return {"played": True}


def _html(csrf_token: str, nonce: str) -> bytes:
    document = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>OneVoice Local Demo</title>
  <style>
    :root{color-scheme:dark;background:#07111f;color:#e8f0fb;font:16px system-ui,sans-serif}
    body{margin:0;background:radial-gradient(circle at top,#17345d,#07111f 55%);min-height:100vh}
    main{max-width:820px;margin:auto;padding:32px 18px 64px}
    h1{margin-bottom:4px}.sub{color:#a9bed8;margin-top:0}.card{background:#0d1c2f;border:1px solid #27415f;border-radius:16px;padding:20px;margin:18px 0;box-shadow:0 12px 35px #0005}
    label{display:block;font-weight:650;margin:12px 0 6px}textarea,select,input,button{font:inherit}textarea,select,input[type=file]{box-sizing:border-box;width:100%;background:#081523;color:#eef6ff;border:1px solid #3a5878;border-radius:9px;padding:11px}textarea{min-height:110px;resize:vertical}
    button{background:#40b6a6;color:#03110f;border:0;border-radius:9px;padding:11px 16px;font-weight:750;cursor:pointer;margin-top:12px}button:disabled{opacity:.5;cursor:not-allowed}button:focus-visible,textarea:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #f1c75b;outline-offset:2px}
    .status{padding:12px;border-radius:9px;background:#122843;white-space:pre-wrap}.ok{border-left:5px solid #40b6a6}.bad{border-left:5px solid #f06b72}.warn{border-left:5px solid #f1c75b}.hidden{display:none}.result{margin-top:14px;white-space:pre-wrap;overflow-wrap:anywhere}code{color:#f1c75b}
    @media(max-width:520px){main{padding:20px 12px}.card{padding:15px}}
  </style>
</head>
<body><main>
  <h1>OneVoice</h1><p class="sub">Demo cục bộ Việt ↔ Anh · Không gửi dữ liệu ra mạng</p>
  <div id="system" class="status warn" role="status" aria-live="polite">Đang kiểm tra pipeline…</div>
  <section class="card" aria-labelledby="text-title"><h2 id="text-title">Dịch văn bản</h2>
    <form id="text-form"><label for="text-lang">Ngôn ngữ nguồn</label><select id="text-lang"><option value="vi">Tiếng Việt</option><option value="en">English</option></select>
    <label for="text-input">Nội dung</label><textarea id="text-input" required maxlength="4096"></textarea><button type="submit">Dịch an toàn</button></form>
    <div id="text-result" class="result" role="status" aria-live="polite"></div>
  </section>
  <section class="card" aria-labelledby="audio-title"><h2 id="audio-title">Dịch file âm thanh</h2>
    <form id="audio-form"><label for="audio-lang">Ngôn ngữ nguồn</label><select id="audio-lang"><option value="">Tự nhận diện</option><option value="vi">Tiếng Việt</option><option value="en">English</option></select>
    <label for="audio-input">Audio tối đa 16 MiB và 30 giây</label><input id="audio-input" type="file" accept="audio/*" required><button type="submit">Nhận dạng và dịch</button></form>
    <div id="audio-result" class="result" role="status" aria-live="polite"></div><button id="play" class="hidden" type="button">Phát audio đã kiểm tra</button>
  </section>
  <p class="sub">Audio không tự phát. Câu hành động lâm sàng luôn cần xác nhận riêng.</p>
</main>
<script nonce="__NONCE__">
const csrf="__CSRF__";let pending=null;
async function api(path,body){const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json","X-OneVoice-CSRF":csrf},body:JSON.stringify(body),cache:"no-store"});const j=await r.json();if(!r.ok)throw new Error(j.error||"request_failed");return j}
function show(el,data){if(!data.safety_passed){el.textContent="ĐÃ CHẶN BỞI CLINICAL SAFETY GATE: "+data.safety_issue_codes.join(", ");el.className="result status bad";return}el.textContent=(data.asr_text?"Nhận dạng: "+data.asr_text+"\n":"")+"Bản dịch: "+data.translated_text+"\nLatency: "+Math.round(data.latency_ms)+" ms"+(data.degraded_mode==="text_only"?"\nText-only fallback: "+data.degradation_code:"");el.className="result status ok"}
document.getElementById("text-form").addEventListener("submit",async e=>{e.preventDefault();const out=document.getElementById("text-result");out.textContent="Đang xử lý…";try{show(out,await api("/api/translate-text",{text:document.getElementById("text-input").value,source_lang:document.getElementById("text-lang").value}))}catch(_){out.textContent="Không thể xử lý yêu cầu.";out.className="result status bad"}});
document.getElementById("audio-form").addEventListener("submit",async e=>{e.preventDefault();const out=document.getElementById("audio-result"),file=document.getElementById("audio-input").files[0],play=document.getElementById("play");play.classList.add("hidden");pending=null;if(!file||file.size>16777216){out.textContent="File audio thiếu hoặc vượt 16 MiB.";out.className="result status bad";return}out.textContent="Đang xử lý…";try{const url=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=reject;r.readAsDataURL(file)}),data=await api("/api/translate-audio",{audio_base64:String(url).split(",",2)[1],source_lang:document.getElementById("audio-lang").value||null});show(out,data);if(data.playback_available){pending=data;play.textContent=data.requires_confirmation?"Xác nhận và phát audio":"Phát audio đã kiểm tra";play.classList.remove("hidden")}}catch(_){out.textContent="Không thể xử lý audio.";out.className="result status bad"}});
document.getElementById("play").addEventListener("click",async e=>{if(!pending)return;e.target.disabled=true;try{await api("/api/playback",{request_id:pending.request_id,confirmed:pending.requires_confirmation});e.target.textContent="Đã phát";pending=null}catch(_){e.target.textContent="Phát thất bại"}finally{e.target.disabled=false}});
fetch("/api/status",{headers:{"X-OneVoice-CSRF":csrf},cache:"no-store"}).then(async r=>{const j=await r.json();const el=document.getElementById("system");el.textContent=j.ready?"Pipeline sẵn sàng · "+j.operational_mode:"Pipeline chưa sẵn sàng";el.className="status "+(j.ready?"ok":"bad")}).catch(()=>{document.getElementById("system").textContent="Không đọc được trạng thái pipeline"});
</script></body></html>"""
    return document.replace("__CSRF__", csrf_token).replace("__NONCE__", nonce).encode("utf-8")


class _LoopbackServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


def create_server(
    pipeline: Any,
    *,
    port: int = DEFAULT_PORT,
    csrf_token: str | None = None,
    playback_store: PendingPlaybackStore | None = None,
) -> ThreadingHTTPServer:
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer in [0, 65535]")
    csrf = secrets.token_urlsafe(32) if csrf_token is None else csrf_token
    if not isinstance(csrf, str) or len(csrf) < 32 or len(csrf) > 256:
        raise ValueError("csrf_token length is invalid")
    store = playback_store or PendingPlaybackStore()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "OneVoice"
        sys_version = ""

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(10.0)

        def log_message(self, _format: str, *_args: object) -> None:
            return

        def send_error(
            self,
            code: int,
            message: str | None = None,
            explain: str | None = None,
        ) -> None:
            self._send_json(code, {"error": "http_error"})

        def _headers(self, content_type: str, length: int) -> None:
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Connection", "close")

        def _send_json(self, status: int, payload: dict[str, object]) -> None:
            encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            self.send_response(status)
            self._headers("application/json; charset=utf-8", len(encoded))
            self.end_headers()
            self.wfile.write(encoded)

        def _valid_host(self) -> bool:
            port_number = self.server.server_address[1]
            return self.headers.get("Host") in {
                f"{LOOPBACK_HOST}:{port_number}",
                f"localhost:{port_number}",
            }

        def _authorized(self) -> bool:
            supplied = self.headers.get("X-OneVoice-CSRF", "")
            return hmac.compare_digest(supplied, csrf)

        def _read_json(self) -> object:
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Transfer-Encoding is unsupported")
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Content-Type must be application/json")
            raw_length = self.headers.get("Content-Length")
            try:
                length = int(raw_length or "")
            except ValueError as error:
                raise ValueError("Content-Length is invalid") from error
            if length < 2 or length > MAX_WEB_REQUEST_BYTES:
                raise ValueError("Request body size is outside the valid range")
            payload = self.rfile.read(length)
            if len(payload) != length:
                raise ValueError("Request body is truncated")
            try:
                return json.loads(payload.decode("utf-8", errors="strict"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("Request body is not valid UTF-8 JSON") from error

        def do_GET(self) -> None:  # noqa: N802
            if not self._valid_host():
                self._send_json(421, {"error": "invalid_host"})
                return
            if self.path == "/":
                nonce = secrets.token_urlsafe(24)
                content = _html(csrf, nonce)
                self.send_response(200)
                self._headers("text/html; charset=utf-8", len(content))
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; connect-src 'self'; img-src 'self' data:; "
                    f"script-src 'nonce-{nonce}'; style-src 'unsafe-inline'; "
                    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
                )
                self.end_headers()
                self.wfile.write(content)
                return
            if self.path == "/api/status":
                if not self._authorized():
                    self._send_json(403, {"error": "forbidden"})
                    return
                try:
                    status = pipeline.get_status()
                except Exception as error:
                    logger.error("Web demo status failure type=%s", type(error).__name__)
                    self._send_json(503, {"error": "pipeline_unavailable"})
                    return
                if not isinstance(status, dict):
                    self._send_json(503, {"error": "pipeline_unavailable"})
                    return
                components = status.get("components", {})
                safe_status = {
                    "ready": status.get("ready") is True,
                    "operational_mode": status.get("operational_mode"),
                    "degradation_code": status.get("degradation_code"),
                    "components": {
                        str(name): ready is True
                        for name, ready in components.items()
                        if isinstance(name, str)
                    }
                    if isinstance(components, dict)
                    else {},
                }
                self._send_json(200, safe_status)
                return
            self._send_json(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._valid_host():
                self._send_json(421, {"error": "invalid_host"})
                return
            if not self._authorized():
                self._send_json(403, {"error": "forbidden"})
                return
            try:
                body = self._read_json()
                if self.path == "/api/translate-text":
                    response = translate_text_payload(pipeline, body)
                elif self.path == "/api/translate-audio":
                    response = translate_audio_payload(pipeline, store, body)
                elif self.path == "/api/playback":
                    response = perform_playback(pipeline, store, body)
                else:
                    self._send_json(404, {"error": "not_found"})
                    return
                self._send_json(200, response)
            except (TypeError, ValueError):
                self._send_json(400, {"error": "invalid_request"})
            except RuntimeError as error:
                logger.error("Web demo runtime failure type=%s", type(error).__name__)
                self._send_json(503, {"error": "pipeline_unavailable"})
            except Exception as error:
                logger.error("Web demo unexpected failure type=%s", type(error).__name__)
                self._send_json(500, {"error": "internal_error"})

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._send_json(405, {"error": "method_not_allowed"})

    return _LoopbackServer((LOOPBACK_HOST, port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OneVoice loopback-only web demo")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)

    from src.pipeline.orchestrator import MediVoicePipeline

    pipeline = MediVoicePipeline(config_path=args.config)
    pipeline.load()
    server = create_server(pipeline, port=args.port)
    logger.info("OneVoice local demo ready at http://%s:%d", *server.server_address)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        return 130
    finally:
        server.server_close()
    return 0


def cli_entrypoint(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"Web demo unavailable ({type(error).__name__}).", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Unexpected web demo failure ({type(error).__name__}).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli_entrypoint())
