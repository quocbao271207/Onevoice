from src.data.quality import RunningProfile, cross_split_leakage, fingerprint_text, normalize_text
from scripts.audit_datasets import duplicate_rate_violations, pair_medev_rows, select_listening_records
from src.training.finetune_mt_medical import MTTrainingConfig, direction_fields, requested_directions
from src.training.finetune_whisper_vi import ASRTrainingConfig, augment_waveform, preflight as asr_preflight
from scripts.audit_mt_alignment import comparison_text
from src.pipeline.safety_guard import validate_translation
from src.pipeline.asr_engine import ASREngine
from src.pipeline.audio_frontend import AudioConfig, AudioFrontend
from src.pipeline.orchestrator import MediVoicePipeline
from src.pipeline.mt_engine import MTEngine, MTResult, NLLB_BASE_REVISION
from src.utils.text_normalization import normalize_for_wer
from scripts.merge_manifests import merge_task, source_records
from scripts.download_datasets import download_dataset, load_registry
from scripts.qc_local_audio import dedupe_exact_audio, repair_dc_offsets
from scripts.apply_listening_review import LANGUAGE_BY_SOURCE
from src.pipeline.flash_cache import FlashCache
from src.pipeline.asr_engine import BASE_MODEL_REVISIONS
from scripts.smoke_cpu_pipeline import build_base_asr, build_base_mt
from scripts.aihub_workbench import (
    build_random_calibration,
    build_profile_options,
    describe_model,
    load_input_specs,
    model_input_specs,
    projected_autoregressive_latency,
    summarize_profile,
)
from scripts.export_phowhisper_aihub import validate_whisper_small_config
from scripts.check_onnx_qnn_compat import find_dynamic_shape_risks
from scripts.prepare_phowhisper_aimet import (
    decoder_input_spec,
    require_linux,
    resolve_model_checkpoint,
    select_calibration_rows,
    teacher_forced_decoder_batches,
    validate_decoder_calibration_batch,
)


def test_normalize_and_fingerprint_cross_source_duplicates():
    assert normalize_text("  Sốc   phản vệ \n") == "Sốc phản vệ"
    assert fingerprint_text("Sốc phản vệ!") == fingerprint_text(" sốc phản vệ ")


def test_profile_reports_duplicates_and_duration():
    profile = RunningProfile()
    profile.add("xin chào", 2.0, {"accent": "North"})
    profile.add("Xin chào!", 2.0, {"accent": "South"})
    result = profile.to_dict()
    assert result["rows"] == 2
    assert result["exact_duplicate_rows"] == 1
    assert result["total_hours"] > 0


def test_medev_parallel_halves_are_paired_by_index():
    pairs, errors = pair_medev_rows(["hello", "pain", "xin chào", "đau"], "train")
    assert not errors
    assert [(x["source_text"], x["target_text"]) for x in pairs] == [
        ("hello", "xin chào"),
        ("pain", "đau"),
    ]


def test_medev_odd_rows_fail_structural_check():
    _, errors = pair_medev_rows(["hello", "pain", "xin chào"], "train")
    assert errors == ["odd_row_count:3"]


def test_cross_split_leakage_detects_text_speaker_and_recording():
    fp = fingerprint_text("bệnh nhân đau ngực")
    result = cross_split_leakage(
        [
            {"role": "train", "text_fingerprint": fp, "speaker": "s1", "group": "g1"},
            {"role": "test", "text_fingerprint": fp, "speaker": "s1", "group": "g1"},
        ]
    )
    assert result["text"]["overlap_count"] == 1
    assert result["speaker"]["overlap_count"] == 1
    assert result["group"]["overlap_count"] == 1
    assert result["speaker"]["role_pair_counts"] == {"test<->train": 1}


def test_translation_direction_is_reversible():
    assert direction_fields("en_to_vi") == ("source_text", "target_text", "eng_Latn", "vie_Latn")
    assert direction_fields("vi_to_en") == ("target_text", "source_text", "vie_Latn", "eng_Latn")
    assert requested_directions("joint") == ("en_to_vi", "vi_to_en")


def test_untranslated_pair_comparison_ignores_spacing_and_punctuation():
    assert comparison_text("Pain Med 8 (4): 326-331.") == comparison_text("Pain Med 8 (4): 326 – 331")


def test_listening_queue_includes_flagged_and_split_coverage():
    rows = [
        {"id": "a", "source_split": "train", "quality_flags": [], "duration_s": 2, "accent": "North"},
        {"id": "b", "source_split": "test", "quality_flags": ["too_long"], "duration_s": 40, "accent": "South"},
    ]
    selected = select_listening_records(rows, limit=2)
    assert selected[0]["id"] == "b"
    assert {row["source_split"] for row in selected} == {"train", "test"}


def test_safety_guard_preserves_numbers_units_and_negation():
    safe = validate_translation("Tiêm 0,5 mg", "Inject 0.5 mg", "vi", "en")
    assert safe.safe
    unsafe = validate_translation("Không tiêm 5 mg", "Inject 50 mg", "vi", "en")
    assert not unsafe.safe
    assert any(issue.startswith("number_mismatch") for issue in unsafe.issues)
    assert "negation_mismatch" in unsafe.issues


def test_safety_guard_accepts_decimal_spacing_number_words_and_implicit_negation():
    decimal = validate_translation("53.8% of 20%", "53, 8% của 20%", "en", "vi")
    assert decimal.safe
    words = validate_translation("13 patients, 6 adults, 6 children", "13 patients, six adults, six children", "vi", "en")
    assert words.safe
    implicit = validate_translation("unequal leg length", "chiều dài chân không đều", "en", "vi")
    assert implicit.safe
    assert validate_translation("e.g. aspirin", "ví dụ aspirin", "en", "vi").safe


def test_safety_guard_rejects_swapped_number_unit_pairs():
    result = validate_translation(
        "Tiêm 5 mg thuốc A và truyền 10 ml thuốc B",
        "Inject 10 mg of drug A and infuse 5 ml of drug B",
        "vi",
        "en",
    )
    assert not result.safe
    assert any(issue.startswith("quantity_mismatch") for issue in result.issues)


def test_safety_guard_normalizes_medical_identifiers_without_hiding_changes():
    equivalent = validate_translation(
        "Chỉ tăng T3 (nhiễm độc T 3) và COVID-19",
        "Only T3 is elevated (T3 toxicosis) and COVID19",
        "vi",
        "en",
    )
    assert equivalent.safe

    changed = validate_translation("Chỉ tăng T3", "Only T4 is elevated", "vi", "en")
    assert not changed.safe
    assert any(issue.startswith("identifier_mismatch") for issue in changed.issues)


def test_audio_config_rounds_silence_detection_up_to_full_chunks():
    config = AudioConfig(chunk_duration_ms=500, silence_duration_ms=800)
    assert config.silence_chunks == 2


def test_pipeline_rejects_invalid_direction_before_cache_lookup():
    pipeline = MediVoicePipeline(config_path="configs/pipeline_config.yaml")
    pipeline._is_loaded = True
    pipeline.flash_cache.load()

    import pytest

    with pytest.raises(ValueError, match="vi_to_vi"):
        pipeline.translate_text("Tiêm epinephrine ngay!", "vi", "vi")


def test_text_only_pipeline_exposes_swapped_drug_dose_safety_failure(monkeypatch):
    pipeline = MediVoicePipeline(config_path="configs/pipeline_config.yaml")
    pipeline._is_loaded = True
    monkeypatch.setattr(pipeline.flash_cache, "lookup", lambda *_: None)
    monkeypatch.setattr(
        pipeline.mt_engine,
        "translate",
        lambda text, source_lang, target_lang: MTResult(
            source_text=text,
            translated_text="Dùng aspirin 10 mg và warfarin 5 mg.",
            source_lang=source_lang,
            target_lang=target_lang,
            latency_ms=1.0,
            first_token_ms=None,
            tokens_generated=10,
            from_cache=False,
        ),
    )

    result = pipeline.translate_text(
        "Give aspirin 5 mg and warfarin 10 mg.",
        "en",
        "vi",
    )

    assert not result.safety_passed
    assert any(
        issue.startswith("quantity_binding_mismatch")
        for issue in result.safety_issues
    )


def test_mt_engine_direct_api_exposes_clinical_safety_failure():
    class FakeInputs(dict):
        def to(self, _device):
            return self

    class FakeTokenizer:
        src_lang = None
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, *_args, **_kwargs):
            import numpy as np

            return FakeInputs(input_ids=np.zeros((1, 3), dtype=np.int64))

        def convert_tokens_to_ids(self, _value):
            return 2

        def decode(self, _tokens, **_kwargs):
            return "Dùng aspirin 10 mg và warfarin 5 mg."

    class FakeModel:
        def generate(self, **_kwargs):
            return [[2, 3, 1]]

    engine = MTEngine(device="cpu")
    engine.tokenizer = FakeTokenizer()
    engine.model = FakeModel()
    engine._is_loaded = True

    result = engine.translate(
        "Give aspirin 5 mg and warfarin 10 mg.",
        "en",
        "vi",
    )

    assert not result.safety_passed
    assert any(
        issue.startswith("quantity_binding_mismatch")
        for issue in result.safety_issues
    )


def test_mt_engine_rejects_blank_and_overlength_source_without_truncation():
    import numpy as np
    import pytest

    class FakeInputs(dict):
        def to(self, _device):
            return self

    class FakeTokenizer:
        src_lang = None
        calls = []

        def __call__(self, *_args, **kwargs):
            self.calls.append(kwargs)
            return FakeInputs(input_ids=np.zeros((1, 5), dtype=np.int64))

    engine = MTEngine(device="cpu", max_source_tokens=4)
    engine.tokenizer = FakeTokenizer()
    engine.model = object()
    engine._is_loaded = True

    with pytest.raises(ValueError, match="must not be blank"):
        engine.translate("   ", "en", "vi")
    assert not engine.tokenizer.calls

    with pytest.raises(ValueError, match="5 tokens.*limit is 4"):
        engine.translate("Give aspirin now.", "en", "vi")
    assert engine.tokenizer.calls == [{"return_tensors": "pt", "truncation": False}]


def test_mt_engine_rejects_generation_that_never_reaches_eos():
    import numpy as np
    import pytest

    class FakeInputs(dict):
        def to(self, _device):
            return self

    class FakeTokenizer:
        src_lang = None
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, *_args, **_kwargs):
            return FakeInputs(input_ids=np.zeros((1, 3), dtype=np.int64))

        def convert_tokens_to_ids(self, _value):
            return 2

        def decode(self, _tokens, **_kwargs):
            raise AssertionError("Incomplete output must not be decoded")

    class FakeModel:
        def generate(self, **_kwargs):
            return [[2, 3, 4]]

    engine = MTEngine(device="cpu")
    engine.tokenizer = FakeTokenizer()
    engine.model = FakeModel()
    engine._is_loaded = True

    with pytest.raises(RuntimeError, match="did not produce EOS"):
        engine.translate("Give aspirin now.", "en", "vi")


def test_text_only_cache_hit_preserves_confirmation_and_safety_metadata():
    pipeline = MediVoicePipeline(config_path="configs/pipeline_config.yaml")
    pipeline._is_loaded = True
    pipeline.flash_cache.load()

    result = pipeline.translate_text("Tiêm epinephrine ngay!", "vi", "en")

    assert result.from_cache
    assert result.safety_passed
    assert result.requires_confirmation


def test_wer_normalizer_preserves_vietnamese_diacritics():
    assert normalize_for_wer("  SỐC phản-vệ! ") == "sốc phản vệ"


def test_asr_resamples_and_downmixes_to_whisper_rate():
    import numpy as np

    stereo = np.zeros((22050, 2), dtype=np.float32)
    audio, sample_rate = ASREngine._to_whisper_rate(stereo, 22050)
    assert sample_rate == 16000
    assert audio.ndim == 1
    assert abs(len(audio) - 16000) <= 1


def test_audio_boundary_rejects_empty_nonfinite_and_unnormalized_input():
    import numpy as np
    import pytest

    invalid = (
        np.array([], dtype=np.float32),
        np.array([0.0, np.nan], dtype=np.float32),
        np.array([0.0, np.inf], dtype=np.float32),
        np.array([0.0, 1.25], dtype=np.float32),
        np.array([[1.25, -1.25]], dtype=np.float32),
    )
    for audio in invalid:
        with pytest.raises(ValueError):
            ASREngine._to_whisper_rate(audio, 16000)
    with pytest.raises(ValueError, match="finite"):
        AudioFrontend().process_chunk(np.array([0.0, np.nan], dtype=np.float32))


def test_audio_boundary_rejects_invalid_rate_shape_and_channel_layout():
    import numpy as np
    import pytest

    mono = np.zeros(160, dtype=np.float32)
    with pytest.raises(ValueError, match="sample_rate"):
        ASREngine._to_whisper_rate(mono, 0)
    with pytest.raises(ValueError, match="sample_rate"):
        ASREngine._to_whisper_rate(mono, 7999)
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        ASREngine._to_whisper_rate(np.zeros((2, 2, 2), dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="channel-last"):
        ASREngine._to_whisper_rate(np.zeros((2, 160), dtype=np.float32), 16000)


def test_audio_config_rejects_invalid_runtime_values():
    import pytest

    with pytest.raises(ValueError, match="sample_rate"):
        AudioConfig(sample_rate=0)
    with pytest.raises(ValueError, match="chunk_duration_ms"):
        AudioConfig(chunk_duration_ms=0)
    with pytest.raises(ValueError, match="vad_threshold"):
        AudioConfig(vad_threshold=1.1)
    with pytest.raises(ValueError, match="vad_threshold"):
        AudioConfig(vad_threshold="0.5")


def test_merge_locks_test_recording_group_before_train(tmp_path):
    import json

    records = tmp_path / "records"
    output = tmp_path / "out"
    records.mkdir()
    spec = {"priority": 1, "train_splits": ["train"], "validation_splits": [], "test_splits": ["test"]}
    test = {"role": "test", "text_fingerprint": "test-fp", "group": "same-video", "quality_flags": []}
    train = {"role": "train", "text_fingerprint": "different-fp", "group": "same-video", "quality_flags": []}
    (records / "sample--test.jsonl").write_text(json.dumps(test) + "\n", encoding="utf-8")
    (records / "sample--train.jsonl").write_text(json.dumps(train) + "\n", encoding="utf-8")

    result = merge_task("asr", [("sample", spec)], records, output)
    assert result["kept"] == {"test": 1}
    assert result["dropped"] == {"group_overlap_with_higher_role": 1}


def test_group_repartition_keeps_locked_hard_video_out_of_train(tmp_path):
    import json

    spec = {
        "split_policy": "group_hash_repartition",
        "repartition_splits": ["train"],
        "locked_test_splits": ["hard"],
        "repartition_ratios": {"train": 1.0, "validation": 0.0, "test": 0.0},
        "repartition_seed": 7,
    }
    train_rows = [
        {"id": "same", "source_split": "train", "group": "locked-video"},
        {"id": "free", "source_split": "train", "group": "free-video"},
    ]
    hard_rows = [{"id": "hard", "source_split": "hard", "group": "locked-video"}]
    for split, rows in (("train", train_rows), ("hard", hard_rows)):
        (tmp_path / f"sample--{split}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
    roles = source_records("sample", spec, tmp_path, "configured")
    assert {row["id"] for row in roles["test"]} == {"same", "hard"}
    assert {row["id"] for row in roles["train"]} == {"free"}


def test_official_split_group_cap_is_deterministic_and_balanced(tmp_path):
    import json

    spec = {
        "train_splits": ["train"],
        "validation_splits": [],
        "test_splits": [],
        "max_train_per_group": 2,
        "sampling_seed": 11,
    }
    rows = [
        {"id": f"a-{index}", "source_split": "train", "group": "speaker-a"}
        for index in range(5)
    ] + [
        {"id": f"b-{index}", "source_split": "train", "group": "speaker-b"}
        for index in range(4)
    ]
    (tmp_path / "sample--train.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    first = source_records("sample", spec, tmp_path, "configured")["train"]
    second = source_records("sample", spec, tmp_path, "configured")["train"]
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) == 4
    assert {row["merge_policy"] for row in first} == {"official_split_group_capped"}


def test_download_registry_contains_only_locked_enabled_sources():
    registry, revisions = load_registry()
    for spec in registry.values():
        if spec.get("enabled", False):
            assert spec["repo_id"] in revisions


def test_snapshot_downloader_refuses_disabled_and_eval_only_sources(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="disabled"):
        download_dataset("futurebee_medical", tmp_path, 1, False)
    with pytest.raises(ValueError, match="evaluation-only"):
        download_dataset("eka_medical_en", tmp_path, 1, False)


def test_exact_audio_dedup_keeps_higher_role_and_logs_removal(tmp_path):
    import json

    audio = tmp_path / "same.flac"
    audio.write_bytes(b"same-audio")
    other = tmp_path / "same-copy.flac"
    other.write_bytes(b"same-audio")
    for role, record_id, path in (
        ("train", "train-id", audio),
        ("test", "test-id", other),
    ):
        row = {
            "id": record_id,
            "source": "sample",
            "source_split": role,
            "role": role,
            "text": role,
            "audio_path": str(path),
            "quality_flags": [],
        }
        (tmp_path / f"asr--{role}-local.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )
    (tmp_path / "asr--validation-local.jsonl").write_text("", encoding="utf-8")

    result = dedupe_exact_audio(tmp_path, ["train", "validation", "test"], workers=2)
    assert result["cross_role_groups"] == 1
    assert result["removed_by_role"] == {"train": 1}
    assert (tmp_path / "asr--train-local.jsonl").read_text(encoding="utf-8") == ""
    kept = json.loads((tmp_path / "asr--test-local.jsonl").read_text(encoding="utf-8"))
    assert kept["id"] == "test-id"
    assert kept["audio_sha256"]


def test_asr_preflight_rejects_exact_audio_hash_leakage(tmp_path):
    import json
    import pytest

    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"exists")
    manifests = []
    for role in ("train", "validation"):
        manifest = tmp_path / f"{role}.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "id": role,
                    "text": role,
                    "text_fingerprint": role,
                    "speaker": f"speaker-{role}",
                    "group": f"group-{role}",
                    "audio_path": str(audio),
                    "audio_sha256": "same-hash",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        manifests.append(manifest)
    config = ASRTrainingConfig(
        train_manifest=str(manifests[0]),
        validation_manifest=str(manifests[1]),
    )
    with pytest.raises(ValueError, match="exact-audio leakage"):
        asr_preflight(config)


def test_vivos_owner_review_is_recorded_as_vietnamese():
    assert LANGUAGE_BY_SOURCE["vivos"] == "vi"


def test_training_defaults_pin_locked_model_revisions():
    import yaml

    lock = yaml.safe_load(open("configs/artifact_lock.yaml", encoding="utf-8"))
    models = lock["base_models"]
    assert models[ASRTrainingConfig.base_model] == ASRTrainingConfig.base_model_revision
    assert models[MTTrainingConfig.base_model] == MTTrainingConfig.base_model_revision


def test_flash_cache_is_exact_by_default_and_flags_clinical_actions():
    cache = FlashCache()
    cache.load()
    action = cache.lookup("Tiêm epinephrine ngay!", "vi")
    assert action is not None
    assert action.requires_confirmation
    assert cache.lookup("xin tiêm epinephrine ngay", "vi") is None


def test_runtime_fallbacks_use_locked_model_revisions():
    import yaml

    models = yaml.safe_load(open("configs/artifact_lock.yaml", encoding="utf-8"))["base_models"]
    assert BASE_MODEL_REVISIONS["vinai/PhoWhisper-small"] == models["vinai/PhoWhisper-small"]
    assert BASE_MODEL_REVISIONS["distil-whisper/distil-small.en"] == models["distil-whisper/distil-small.en"]
    assert NLLB_BASE_REVISION == models["facebook/nllb-200-distilled-600M"]


def test_runtime_base_fallback_is_explicitly_pre_gpu_only():
    import yaml

    pipeline = yaml.safe_load(open("configs/pipeline_config.yaml", encoding="utf-8"))
    assert pipeline["runtime"]["allow_base_model_fallback"] is True
    assert ASREngine().allow_base_fallback is False


def test_runtime_mt_decoding_matches_selected_validation_setting():
    import yaml

    pipeline = yaml.safe_load(open("configs/pipeline_config.yaml", encoding="utf-8"))
    assert pipeline["mt"]["num_beams"] == 1
    assert pipeline["mt"]["max_source_tokens"] == 256
    assert MTEngine().num_beams == 1
    assert MTEngine().max_source_tokens == 256


def test_cpu_smoke_uses_fail_closed_engine_defaults():
    asr = build_base_asr()
    mt = build_base_mt()
    assert asr.vi_model_path == "vinai/PhoWhisper-small"
    assert asr.en_model_path == "distil-whisper/distil-small.en"
    assert mt.model_path == "facebook/nllb-200-distilled-600M"
    assert not asr.allow_base_fallback
    assert not mt.allow_base_fallback


def test_duplicate_rate_gate_uses_declared_threshold():
    profile = RunningProfile()
    profile.add("same")
    profile.add("same")
    assert duplicate_rate_violations({"train": profile}, 0.40) == {"train": 0.5}
    assert duplicate_rate_violations({"train": profile}, 0.50) == {}


def test_mt_context_defaults_match_bidirectionally():
    config = MTTrainingConfig()
    assert config.max_source_length == 256
    assert config.max_target_length == 256


def test_training_clis_expose_low_vram_batch_controls():
    import subprocess
    import sys

    for module in (
        "src.training.finetune_whisper_vi",
        "src.training.finetune_mt_medical",
    ):
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            capture_output=True,
            check=True,
            text=True,
        )
        assert "--eval-batch-size" in result.stdout
        assert "--gradient-accumulation-steps" in result.stdout


def test_aihub_autoregressive_latency_projection_counts_every_token():
    projected = projected_autoregressive_latency(233.30, 23.46, [1, 20, 200])
    assert projected == {"1": 256.76, "20": 702.5, "200": 4925.3}


def test_aihub_profile_options_are_explicit_and_fail_closed():
    assert build_profile_options("2.50", "npu") == " --qairt_version 2.50 --compute_unit npu"

    import pytest

    with pytest.raises(ValueError, match="unsupported compute unit"):
        build_profile_options("2.50", "auto")


def test_aihub_profile_summary_keeps_metrics_and_compute_coverage():
    result = summarize_profile(
        {
            "execution_summary": {
                "estimated_inference_time": 22954,
                "all_inference_times": [1000, 2000, 3000, 4000],
                "all_first_load_times": [5000],
                "all_warm_load_times": [],
            },
            "execution_detail": [
                {"compute_unit": "NPU"},
                {"compute_unit": "NPU"},
                {"compute_unit": "CPU"},
            ],
        }
    )
    assert result == {
        "execution_summary": {"estimated_inference_time": 22954},
        "latency_distribution": {
            "inference": {
                "samples": 4,
                "min_ms": 1.0,
                "p50_ms": 2.5,
                "p95_ms": 4.0,
                "max_ms": 4.0,
            },
            "first_load": {
                "samples": 1,
                "min_ms": 5.0,
                "p50_ms": 5.0,
                "p95_ms": 5.0,
                "max_ms": 5.0,
            },
            "warm_load": {"samples": 0},
        },
        "operation_count_by_compute_unit": {"NPU": 2, "CPU": 1},
    }


def test_aihub_model_evidence_serializes_enum_like_values():
    from types import SimpleNamespace

    evidence = describe_model(
        SimpleNamespace(model_id="m123", name="encoder.onnx", model_type=object())
    )
    assert evidence["model_id"] == "m123"
    assert evidence["name"] == "encoder.onnx"
    assert isinstance(evidence["model_type"], str)


def test_aihub_model_input_specs_flattens_graph_groups():
    from types import SimpleNamespace

    model = SimpleNamespace(
        input_spec={
            None: [
                SimpleNamespace(name="input", shape=(1, 80, 3000), dtype="float32"),
                SimpleNamespace(name="position", shape=(1,), dtype="int32"),
            ]
        }
    )
    assert model_input_specs(model) == {
        "input": ((1, 80, 3000), "float32"),
        "position": ((1,), "int32"),
    }


def test_aihub_input_specs_json_is_validated(tmp_path):
    import json
    import pytest

    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps(
            {
                "input": {"shape": [1, 128], "dtype": "int64"},
                "scales": {"shape": [3], "dtype": "float32"},
            }
        ),
        encoding="utf-8",
    )
    assert load_input_specs(valid) == {
        "input": ((1, 128), "int64"),
        "scales": ((3,), "float32"),
    }

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps({"input": {"shape": [1, 0], "dtype": "int64"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid shape"):
        load_input_specs(invalid)


def test_aihub_random_calibration_is_bounded_and_preserves_dtypes():
    calibration, total_bytes = build_random_calibration(
        {
            "features": ((1, 2, 3), "float32"),
            "tokens": ((1, 2), "int32"),
        },
        sample_count=1,
        seed=7,
        max_bytes=64,
    )
    assert total_bytes == 32
    assert calibration["features"][0].dtype.name == "float32"
    assert calibration["tokens"][0].dtype.name == "int32"
    assert calibration["tokens"][0].sum() == 0

    import pytest

    with pytest.raises(ValueError, match="exceeds"):
        build_random_calibration(
            {"features": ((1, 80, 3000), "float32")},
            sample_count=1,
            seed=7,
            max_bytes=1024,
        )


def test_aihub_random_calibration_uses_valid_sequence_smoke_values():
    calibration, _ = build_random_calibration(
        {
            "input": ((1, 8), "int64"),
            "input_lengths": ((1,), "int64"),
            "scales": ((3,), "float32"),
            "sid": ((1,), "int64"),
        },
        sample_count=1,
        seed=7,
    )
    assert calibration["input"][0].min() >= 1
    assert calibration["input_lengths"][0].tolist() == [8]
    assert calibration["scales"][0].tolist() == [1.0, 1.0, 1.0]
    assert calibration["sid"][0].tolist() == [0]


def test_qnn_compat_check_detects_data_dependent_range_and_nonzero():
    from types import SimpleNamespace

    nodes = [
        SimpleNamespace(
            name="/ReduceMax",
            op_type="ReduceMax",
            input=["durations"],
            output=["max_duration"],
        ),
        SimpleNamespace(
            name="/Range",
            op_type="Range",
            input=["zero", "max_duration", "one"],
            output=["time_axis"],
        ),
        SimpleNamespace(
            name="/NonZero",
            op_type="NonZero",
            input=["mask"],
            output=["indices"],
        ),
    ]
    risks = find_dynamic_shape_risks(nodes, {"zero", "one"})
    assert [risk["node"] for risk in risks] == ["/NonZero", "/Range"]
    assert [risk["severity"] for risk in risks] == ["block", "block"]
    assert risks[1]["limit_producer"] == "ReduceMax"


def test_phowhisper_aimet_handoff_fails_closed_off_linux():
    import pytest

    require_linux("Linux")
    with pytest.raises(RuntimeError, match="requires Linux"):
        require_linux("Windows")


def test_phowhisper_calibration_selection_is_train_only_and_deterministic(tmp_path):
    import json
    import pytest

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rows = []
    for index in range(3):
        audio = audio_dir / f"{index}.flac"
        audio.write_bytes(b"audio")
        rows.append(
            {
                "id": f"row-{index}",
                "role": "train",
                "audio_path": f"audio/{index}.flac",
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    first = select_calibration_rows(manifest, tmp_path, limit=2, seed=7)
    second = select_calibration_rows(manifest, tmp_path, limit=2, seed=7)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) == 2

    rows[0]["role"] = "test"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-train row"):
        select_calibration_rows(manifest, tmp_path, limit=2, seed=7)


def test_phowhisper_aimet_handoff_rejects_missing_local_checkpoint(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        resolve_model_checkpoint(
            tmp_path / "missing",
            "vinai/PhoWhisper-small",
            "revision",
        )


def test_phowhisper_decoder_calibration_contract_matches_qualcomm_recipe():
    from types import SimpleNamespace

    import numpy as np

    config = SimpleNamespace(
        decoder_layers=2,
        decoder_attention_heads=4,
        d_model=16,
    )
    specs = decoder_input_spec(config, mean_decode_len=5, audio_emb_len=3)
    assert list(specs) == [
        "input_ids",
        "attention_mask",
        "k_cache_self_0_in",
        "v_cache_self_0_in",
        "k_cache_self_1_in",
        "v_cache_self_1_in",
        "k_cache_cross_0",
        "v_cache_cross_0",
        "k_cache_cross_1",
        "v_cache_cross_1",
        "position_ids",
    ]
    batch = {
        name: np.zeros(shape, dtype=dtype)
        for name, (shape, dtype) in specs.items()
    }
    validate_decoder_calibration_batch(batch, specs)


def test_phowhisper_decoder_calibration_contract_rejects_wrong_shape_and_dtype():
    from types import SimpleNamespace

    import numpy as np
    import pytest

    specs = decoder_input_spec(
        SimpleNamespace(decoder_layers=1, decoder_attention_heads=2, d_model=8),
        mean_decode_len=4,
        audio_emb_len=3,
    )
    batch = {
        name: np.zeros(shape, dtype=dtype)
        for name, (shape, dtype) in specs.items()
    }
    batch["position_ids"] = np.zeros((1,), dtype="int64")
    with pytest.raises(ValueError, match="position_ids dtype"):
        validate_decoder_calibration_batch(batch, specs)

    batch["position_ids"] = np.zeros((1,), dtype="int32")
    batch["attention_mask"] = np.zeros((1, 1, 1, 3), dtype="float32")
    with pytest.raises(ValueError, match="attention_mask shape"):
        validate_decoder_calibration_batch(batch, specs)


def test_phowhisper_teacher_forcing_streams_real_states_in_memory(
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace

    import numpy as np
    import soundfile as sf
    import torch
    import transformers

    audio = tmp_path / "train.flac"
    sf.write(audio, np.zeros(1600, dtype="float32"), 16000)

    class FakeExtractor:
        @classmethod
        def from_pretrained(cls, _checkpoint):
            return cls()

        def __call__(self, _audio, sampling_rate, return_tensors):
            assert sampling_rate == 16000
            assert return_tensors == "pt"
            return SimpleNamespace(input_features=torch.zeros((1, 80, 3000)))

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, _checkpoint, language, task):
            assert language == "vi"
            assert task == "transcribe"
            return cls()

        def __call__(self, text, return_tensors):
            assert text == "xin chao"
            assert return_tensors == "pt"
            return SimpleNamespace(input_ids=torch.tensor([[11, 12, 13]]))

    monkeypatch.setattr(transformers, "WhisperFeatureExtractor", FakeExtractor)
    monkeypatch.setattr(transformers, "WhisperTokenizer", FakeTokenizer)
    config = SimpleNamespace(
        decoder_layers=1,
        decoder_attention_heads=2,
        d_model=4,
        mask_neg=-100.0,
    )

    def encoder(_features):
        return ((torch.zeros((2, 1, 2, 1500)), torch.zeros((2, 1, 1500, 2))),)

    def decoder(*args):
        next_cache = ((torch.ones_like(args[2]), torch.ones_like(args[3])),)
        return torch.zeros((1, 10, 1, 1)), next_cache

    stats = {}
    batches = list(
        teacher_forced_decoder_batches(
            [
                {
                    "id": "safe-row-id",
                    "text": "xin chao",
                    "resolved_audio_path": str(audio),
                }
            ],
            tmp_path,
            encoder,
            decoder,
            config,
            steps_per_row=2,
            stats=stats,
        )
    )
    assert stats == {"batches": 2}
    assert [int(batch["input_ids"][0, 0]) for batch in batches] == [11, 12]
    assert np.count_nonzero(batches[0]["attention_mask"] == 0) == 1
    assert np.count_nonzero(batches[1]["attention_mask"] == 0) == 2
    assert np.all(batches[0]["k_cache_self_0_in"] == 0)
    assert np.all(batches[1]["k_cache_self_0_in"] == 1)


def test_phowhisper_aihub_export_rejects_non_small_architecture():
    from types import SimpleNamespace

    valid = SimpleNamespace(
        model_type="whisper",
        d_model=768,
        encoder_layers=12,
        decoder_layers=12,
        encoder_attention_heads=12,
        decoder_attention_heads=12,
        num_mel_bins=80,
    )
    validate_whisper_small_config(valid)
    valid.d_model = 512

    import pytest

    with pytest.raises(RuntimeError, match="not Whisper-small compatible"):
        validate_whisper_small_config(valid)


def test_mt_token_report_is_checksum_locked():
    import hashlib
    from pathlib import Path

    import yaml

    report = Path("data/reports/eda/medev_token_lengths.json")
    locked = yaml.safe_load(open("configs/artifact_lock.yaml", encoding="utf-8"))["manifests"]
    assert hashlib.sha256(report.read_bytes()).hexdigest() == locked["mt_token_length_report_sha256"]


def test_alignment_report_has_deterministic_flag_order():
    import json
    from pathlib import Path

    report = json.loads(Path("data/reports/eda/medev_alignment.json").read_text(encoding="utf-8"))
    for split in report["splits"].values():
        assert list(split["flag_counts"]) == sorted(split["flag_counts"])
        assert list(split["examples"]) == sorted(split["examples"])


def test_dc_repair_preserves_clip_and_removes_bias(tmp_path):
    import json
    import numpy as np
    import soundfile as sf

    audio = tmp_path / "biased.flac"
    signal = (0.2 + 0.05 * np.sin(np.linspace(0, 40, 16000))).astype(np.float32)
    sf.write(audio, signal, 16000, format="FLAC", subtype="PCM_16")
    row = {
        "id": "biased",
        "audio_path": str(audio),
        "audio_qc": {"dc_offset": 0.2, "warnings": ["high_dc_offset"]},
    }
    for role in ("train", "validation", "test"):
        rows = [row] if role == "train" else []
        (tmp_path / f"asr--{role}-local.jsonl").write_text(
            "".join(json.dumps(value) + "\n" for value in rows), encoding="utf-8"
        )
    result = repair_dc_offsets(tmp_path, ["train", "validation", "test"])
    repaired, _ = sf.read(audio, dtype="float32")
    assert result["repaired"] == 1
    assert abs(float(np.mean(repaired))) < 1e-4
    assert float(np.std(repaired)) > 0.01


def test_asr_augmentation_is_deterministic_finite_and_nonempty():
    import numpy as np

    config = ASRTrainingConfig(
        speed_probability=1.0,
        gain_probability=1.0,
        noise_probability=1.0,
        reverb_probability=1.0,
        telephone_probability=1.0,
        codec_probability=1.0,
    )
    audio = np.sin(np.linspace(0, 100, 32000)).astype(np.float32) * 0.1
    first = augment_waveform(audio, np.random.default_rng(7), config)
    second = augment_waveform(audio, np.random.default_rng(7), config)
    assert len(first) > 1600
    assert np.isfinite(first).all()
    assert np.max(np.abs(first)) <= 1.0
    assert np.array_equal(first, second)
    assert not np.array_equal(first[: min(len(first), len(audio))], audio[: min(len(first), len(audio))])
