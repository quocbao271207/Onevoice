"""Capture raw runtime telemetry on a verified physical QCS6490 board.

The benchmark command is executed directly without a shell.  Every measured
run records wall-clock latency while power and thermal sysfs sensors are
sampled throughout the process lifetime.  The immutable JSON output is the
only raw format accepted by ``seal_qcs6490_measurement.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.capture_qcs6490_identity import capture_identity  # noqa: E402
from scripts.run_model_bakeoff import (  # noqa: E402
    MAX_DEPLOYMENT_MEASUREMENT_BYTES,
)


MIN_MEASUREMENT_RUNS = 30
MAX_MEASUREMENT_RUNS = 10_000
MAX_WARMUP_RUNS = 1_000
MIN_SAMPLE_INTERVAL_MS = 50
MAX_SAMPLE_INTERVAL_MS = 5_000
MAX_RUN_TIMEOUT_SECONDS = 3_600.0
MAX_MINIMUM_DURATION_SECONDS = 3_600.0
MAX_COMMAND_ARGUMENTS = 256
MAX_COMMAND_ARGUMENT_BYTES = 4_096
MAX_COMMAND_BYTES = 32_768
MAX_SENSOR_BYTES = 4_096
MAX_SENSOR_SAMPLES_PER_CHANNEL = 100_000
MAX_POWER_MW = 1_000_000.0
MAX_TEMPERATURE_C = 250.0
MAX_LATENCY_MS = MAX_RUN_TIMEOUT_SECONDS * 1_000.0
PROCESS_TERMINATION_GRACE_SECONDS = 5.0


@dataclass(frozen=True)
class RunSample:
    """Telemetry captured for one benchmark command invocation."""

    latency_ms: float
    power_samples_mw: tuple[float, ...]
    temperature_samples_c: tuple[float, ...]
    peak_ram_bytes: int


def _finite_number(
    value: object,
    *,
    label: str,
    minimum_exclusive: float,
    maximum_inclusive: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if (
        not math.isfinite(parsed)
        or parsed <= minimum_exclusive
        or (maximum_inclusive is not None and parsed > maximum_inclusive)
    ):
        raise ValueError(f"{label} is outside the valid range")
    return parsed


def _bounded_integer(
    value: object,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return value


def normalize_command(command: Sequence[str]) -> tuple[str, ...]:
    """Validate an argv vector without interpreting shell syntax."""
    normalized = list(command)
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]
    if not normalized:
        raise ValueError("benchmark command is required after --")
    if len(normalized) > MAX_COMMAND_ARGUMENTS:
        raise ValueError("benchmark command has too many arguments")
    total_bytes = 0
    for argument in normalized:
        if not isinstance(argument, str) or not argument or "\x00" in argument:
            raise ValueError("benchmark command contains an invalid argument")
        argument_bytes = len(argument.encode("utf-8"))
        if argument_bytes > MAX_COMMAND_ARGUMENT_BYTES:
            raise ValueError("benchmark command argument is too long")
        total_bytes += argument_bytes
    if total_bytes > MAX_COMMAND_BYTES:
        raise ValueError("benchmark command is too long")
    return tuple(normalized)


def read_scaled_sensor(path: Path, *, scale: float, label: str) -> float:
    """Read one bounded numeric sysfs value and convert it to canonical units."""
    parsed_scale = _finite_number(
        scale,
        label=f"{label} scale",
        minimum_exclusive=0.0,
    )
    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            raw = handle.read(MAX_SENSOR_BYTES + 1)
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Unable to read {label} sensor: {path}") from exc
    if len(raw.encode("utf-8")) > MAX_SENSOR_BYTES:
        raise ValueError(f"{label} sensor value is too large")
    value = _finite_number(
        raw.strip(),
        label=f"{label} sensor value",
        minimum_exclusive=-math.inf,
    )
    converted = value * parsed_scale
    minimum = -273.15 if label == "temperature" else 0.0
    maximum = MAX_TEMPERATURE_C if label == "temperature" else MAX_POWER_MW
    return _finite_number(
        converted,
        label=f"scaled {label} sensor value",
        minimum_exclusive=minimum,
        maximum_inclusive=maximum,
    )


def _validated_sysfs_sensor_path(path: Path, *, label: str) -> Path:
    """Resolve one real sysfs sensor without permitting arbitrary host files."""
    if not path.is_absolute():
        raise ValueError(f"{label} sensor path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} sensor path does not exist: {path}") from exc
    try:
        resolved.relative_to(Path("/sys"))
    except ValueError as exc:
        raise ValueError(f"{label} sensor path must resolve below /sys") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} sensor path is not a readable sysfs file")
    sensor_label = f"sysfs:{path}"
    if len(sensor_label) > 512:
        raise ValueError(f"{label} sensor path is too long")
    return resolved


def _process_group_rss_bytes(process_group_id: int, proc_root: Path) -> int:
    """Return current aggregate RSS for processes still in one process group."""
    total_kib = 0
    try:
        entries = tuple(proc_root.iterdir())
    except OSError:
        return 0
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            close_parenthesis = stat.rfind(")")
            if close_parenthesis < 0:
                continue
            fields = stat[close_parenthesis + 2 :].split()
            if len(fields) < 3 or int(fields[2]) != process_group_id:
                continue
            status = (entry / "status").read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        for line in status.splitlines():
            if not line.startswith("VmRSS:"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    total_kib += int(parts[1])
                except ValueError:
                    pass
            break
    return total_kib * 1024


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)


def run_command_once(
    command: Sequence[str],
    *,
    power_sensor_path: Path,
    power_scale_to_mw: float,
    temperature_sensor_path: Path,
    temperature_scale_to_c: float,
    sample_interval_seconds: float,
    timeout_seconds: float,
    proc_root: Path = Path("/proc"),
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> RunSample:
    """Run one benchmark command and collect live telemetry until it exits."""
    if platform.system() != "Linux":
        raise RuntimeError("QCS6490 runtime capture requires Linux")
    argv = normalize_command(command)
    power_sensor_path = _validated_sysfs_sensor_path(
        power_sensor_path,
        label="power",
    )
    temperature_sensor_path = _validated_sysfs_sensor_path(
        temperature_sensor_path,
        label="temperature",
    )
    started = monotonic()
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=False,
        start_new_session=True,
    )
    power_samples: list[float] = []
    temperature_samples: list[float] = []
    peak_ram_bytes = 0
    try:
        while True:
            power_samples.append(
                read_scaled_sensor(
                    power_sensor_path,
                    scale=power_scale_to_mw,
                    label="power",
                )
            )
            temperature_samples.append(
                read_scaled_sensor(
                    temperature_sensor_path,
                    scale=temperature_scale_to_c,
                    label="temperature",
                )
            )
            peak_ram_bytes = max(
                peak_ram_bytes,
                _process_group_rss_bytes(process.pid, proc_root),
            )
            return_code = process.poll()
            elapsed = monotonic() - started
            if return_code is not None:
                break
            if elapsed >= timeout_seconds:
                raise TimeoutError(
                    f"benchmark command exceeded {timeout_seconds:g} seconds"
                )
            sleep(min(sample_interval_seconds, max(timeout_seconds - elapsed, 0.0)))
    except BaseException:
        _terminate_process_group(process)
        raise

    latency_ms = (monotonic() - started) * 1_000.0
    if return_code != 0:
        raise RuntimeError(f"benchmark command failed with exit code {return_code}")
    if peak_ram_bytes <= 0:
        raise RuntimeError("Unable to observe positive process-group RSS")
    return RunSample(
        latency_ms=latency_ms,
        power_samples_mw=tuple(power_samples),
        temperature_samples_c=tuple(temperature_samples),
        peak_ram_bytes=peak_ram_bytes,
    )


def _validated_run_sample(sample: RunSample) -> RunSample:
    if not isinstance(sample, RunSample):
        raise TypeError("sample runner must return RunSample")
    latency = _finite_number(
        sample.latency_ms,
        label="latency_ms",
        minimum_exclusive=0.0,
        maximum_inclusive=MAX_LATENCY_MS,
    )
    if not sample.power_samples_mw or not sample.temperature_samples_c:
        raise ValueError("each run must include power and temperature samples")
    powers = tuple(
        _finite_number(
            value,
            label="power sample",
            minimum_exclusive=0.0,
            maximum_inclusive=MAX_POWER_MW,
        )
        for value in sample.power_samples_mw
    )
    temperatures = tuple(
        _finite_number(
            value,
            label="temperature sample",
            minimum_exclusive=-273.15,
            maximum_inclusive=MAX_TEMPERATURE_C,
        )
        for value in sample.temperature_samples_c
    )
    ram = _bounded_integer(
        sample.peak_ram_bytes,
        label="peak_ram_bytes",
        minimum=1,
        maximum=sys.maxsize,
    )
    return RunSample(latency, powers, temperatures, ram)


def capture_runtime(
    *,
    command: Sequence[str],
    power_sensor_path: Path,
    power_scale_to_mw: float,
    temperature_sensor_path: Path,
    temperature_scale_to_c: float,
    runs: int = MIN_MEASUREMENT_RUNS,
    warmup_runs: int = 3,
    sample_interval_ms: int = 100,
    timeout_seconds: float = 300.0,
    minimum_duration_seconds: float = 0.0,
    system_root: Path = Path("/"),
    architecture: str | None = None,
    sample_runner: Callable[[], RunSample] | None = None,
    now: Callable[[], datetime] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Capture warm and measured runs after fail-closed physical-board checks."""
    argv = normalize_command(command)
    minimum_runs = _bounded_integer(
        runs,
        label="runs",
        minimum=MIN_MEASUREMENT_RUNS,
        maximum=MAX_MEASUREMENT_RUNS,
    )
    warmups = _bounded_integer(
        warmup_runs,
        label="warmup_runs",
        minimum=0,
        maximum=MAX_WARMUP_RUNS,
    )
    interval_ms = _bounded_integer(
        sample_interval_ms,
        label="sample_interval_ms",
        minimum=MIN_SAMPLE_INTERVAL_MS,
        maximum=MAX_SAMPLE_INTERVAL_MS,
    )
    timeout = _finite_number(
        timeout_seconds,
        label="timeout_seconds",
        minimum_exclusive=0.0,
    )
    if timeout > MAX_RUN_TIMEOUT_SECONDS:
        raise ValueError(
            f"timeout_seconds must be at most {MAX_RUN_TIMEOUT_SECONDS:g}"
        )
    if isinstance(minimum_duration_seconds, bool):
        raise ValueError("minimum_duration_seconds must be a finite number")
    try:
        minimum_duration = float(minimum_duration_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("minimum_duration_seconds must be a finite number") from exc
    if (
        not math.isfinite(minimum_duration)
        or minimum_duration < 0.0
        or minimum_duration > MAX_MINIMUM_DURATION_SECONDS
    ):
        raise ValueError(
            "minimum_duration_seconds must be between 0 and "
            f"{MAX_MINIMUM_DURATION_SECONDS:g}"
        )
    power_scale = _finite_number(
        power_scale_to_mw,
        label="power_scale_to_mw",
        minimum_exclusive=0.0,
    )
    temperature_scale = _finite_number(
        temperature_scale_to_c,
        label="temperature_scale_to_c",
        minimum_exclusive=0.0,
    )

    if system_root == Path("/") and platform.system() != "Linux":
        raise RuntimeError("QCS6490 runtime capture requires Linux")
    capture_identity(system_root, architecture=architecture)
    captured_at = (now or (lambda: datetime.now(timezone.utc)))()
    if captured_at.tzinfo is None:
        raise ValueError("capture timestamp must include a timezone")

    if sample_runner is None:
        sample_runner = lambda: run_command_once(
            argv,
            power_sensor_path=power_sensor_path,
            power_scale_to_mw=power_scale,
            temperature_sensor_path=temperature_sensor_path,
            temperature_scale_to_c=temperature_scale,
            sample_interval_seconds=interval_ms / 1_000.0,
            timeout_seconds=timeout,
        )

    for _ in range(warmups):
        _validated_run_sample(sample_runner())

    latency_samples: list[float] = []
    power_samples: list[float] = []
    temperature_samples: list[float] = []
    peak_ram_bytes = 0
    measurement_started = monotonic()
    while (
        len(latency_samples) < minimum_runs
        or monotonic() - measurement_started < minimum_duration
    ):
        if len(latency_samples) >= MAX_MEASUREMENT_RUNS:
            raise RuntimeError(
                "minimum duration was not reached before the measurement-run limit"
            )
        sample = _validated_run_sample(sample_runner())
        latency_samples.append(sample.latency_ms)
        power_samples.extend(sample.power_samples_mw)
        temperature_samples.extend(sample.temperature_samples_c)
        if (
            len(power_samples) > MAX_SENSOR_SAMPLES_PER_CHANNEL
            or len(temperature_samples) > MAX_SENSOR_SAMPLES_PER_CHANNEL
        ):
            raise RuntimeError("runtime sensor sample limit exceeded")
        peak_ram_bytes = max(peak_ram_bytes, sample.peak_ram_bytes)

    return {
        "version": 1,
        "capture_source": "qcs6490_runtime_sampler",
        "captured_at": captured_at.isoformat(),
        "latency_samples_ms": latency_samples,
        "power_sensor": f"sysfs:{power_sensor_path}",
        "power_samples_mw": power_samples,
        "temperature_sensor": f"sysfs:{temperature_sensor_path}",
        "temperature_samples_c": temperature_samples,
        "peak_ram_bytes": peak_ram_bytes,
        "peak_vram_bytes": 0,
    }


def write_exclusive(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite raw measurement: {path}")
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if len(serialized.encode("utf-8")) > MAX_DEPLOYMENT_MEASUREMENT_BYTES:
        raise ValueError("Raw measurement exceeds the deployment evidence size limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-sensor-path", type=Path, required=True)
    parser.add_argument("--power-scale-to-mw", type=float, required=True)
    parser.add_argument("--temperature-sensor-path", type=Path, required=True)
    parser.add_argument("--temperature-scale-to-c", type=float, required=True)
    parser.add_argument("--runs", type=int, default=MIN_MEASUREMENT_RUNS)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--minimum-duration-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    payload = capture_runtime(
        command=args.command,
        power_sensor_path=args.power_sensor_path,
        power_scale_to_mw=args.power_scale_to_mw,
        temperature_sensor_path=args.temperature_sensor_path,
        temperature_scale_to_c=args.temperature_scale_to_c,
        runs=args.runs,
        warmup_runs=args.warmup_runs,
        sample_interval_ms=args.sample_interval_ms,
        timeout_seconds=args.timeout_seconds,
        minimum_duration_seconds=args.minimum_duration_seconds,
    )
    write_exclusive(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "latency_samples": len(payload["latency_samples_ms"]),
                "power_samples": len(payload["power_samples_mw"]),
                "temperature_samples": len(payload["temperature_samples_c"]),
                "peak_ram_bytes": payload["peak_ram_bytes"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
