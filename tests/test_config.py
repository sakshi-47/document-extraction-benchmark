"""Tests for run configuration, hardware detection and token handling."""

from __future__ import annotations

import pytest
import yaml

from docbench.auth import MissingTokenError, Token, get_token
from docbench.hardware import Device, detect_device
from docbench.train.config import RunConfig, load_run_config
from docbench.train.loop import check_quantization, resolve_precision

MINIMAL = {"name": "t", "base_model": "some/model"}


def write(tmp_path, data):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


class TestRunConfig:
    def test_minimal_config_loads(self, tmp_path):
        assert load_run_config(write(tmp_path, MINIMAL)).name == "t"

    def test_typo_in_a_key_is_an_error(self, tmp_path):
        """A silently-ignored typo means the run is not the run the file describes."""
        with pytest.raises(ValueError, match="learing_rate"):
            load_run_config(write(tmp_path, MINIMAL | {"learing_rate": 1e-4}))

    def test_non_mapping_rejected(self, tmp_path):
        path = tmp_path / "run.yaml"
        path.write_text("- just\n- a list\n")
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            load_run_config(path)

    def test_oversized_config_refused(self, tmp_path):
        path = tmp_path / "run.yaml"
        path.write_text("name: t\nbase_model: m\n# " + "x" * (256 * 1024))
        with pytest.raises(ValueError, match="implausibly large"):
            load_run_config(path)

    def test_effective_batch_size(self):
        cfg = RunConfig(**MINIMAL, batch_size=2, grad_accum_steps=8)
        assert cfg.effective_batch_size == 16

    def test_push_to_hub_requires_a_repo_id(self, tmp_path):
        with pytest.raises(ValueError, match="hub_repo_id is empty"):
            load_run_config(write(tmp_path, MINIMAL | {"checkpoint": {"push_to_hub": True}}))

    @pytest.mark.parametrize(
        "field,value", [("max_steps", 0), ("learning_rate", 0), ("batch_size", 0)]
    )
    def test_nonsense_values_rejected(self, field, value):
        with pytest.raises(ValueError):
            RunConfig(**MINIMAL | {field: value})

    def test_shipped_configs_are_valid(self):
        """The repo's own configs must load, or CI is lying about the run."""
        from pathlib import Path

        for name in ("smoke.yaml", "qwen2vl_lora.yaml"):
            path = Path(__file__).parent.parent / "configs" / name
            assert load_run_config(path).name


class TestPrecision:
    def test_auto_follows_the_device(self):
        cfg = RunConfig(**MINIMAL, precision="auto")
        assert resolve_precision(cfg) == detect_device().torch_dtype

    def test_explicit_bf16_on_pre_ampere_is_an_error(self, monkeypatch):
        pascal = Device("cuda", "Tesla P100", (6, 0), 16.0, supports_bf16=False)
        monkeypatch.setattr("docbench.train.loop.detect_device", lambda: pascal)
        with pytest.raises(ValueError, match="pre-Ampere"):
            resolve_precision(RunConfig(**MINIMAL, precision="bfloat16"))

    def test_pre_ampere_device_picks_fp16_and_scaling(self):
        pascal = Device("cuda", "Tesla P100", (6, 0), 16.0, supports_bf16=False)
        assert pascal.torch_dtype == "float16"
        assert pascal.needs_grad_scaling

    def test_ampere_device_picks_bf16_without_scaling(self):
        ampere = Device("cuda", "A100", (8, 0), 40.0, supports_bf16=True)
        assert ampere.torch_dtype == "bfloat16"
        assert not ampere.needs_grad_scaling

    def test_pascal_cannot_run_4bit(self):
        """P100 is below the bitsandbytes 4-bit floor; QLoRA cannot run on it."""
        pascal = Device("cuda", "Tesla P100-PCIE-16GB", (6, 0), 16.0, supports_bf16=False)
        assert not pascal.supports_int4
        assert "no 4-bit" in pascal.describe()

    def test_turing_can_run_4bit(self):
        turing = Device("cuda", "Tesla T4", (7, 5), 15.8, supports_bf16=False)
        assert turing.supports_int4
        assert turing.torch_dtype == "float16", "Turing has no bf16 either"
        assert "4-bit ok" in turing.describe()

    def test_cpu_is_never_4bit_capable(self):
        assert not Device("cpu", "cpu", None, None, False).supports_int4

    def test_4bit_config_on_pascal_fails_before_training(self, monkeypatch):
        pascal = Device("cuda", "Tesla P100-PCIE-16GB", (6, 0), 16.0, supports_bf16=False)
        monkeypatch.setattr("docbench.train.loop.detect_device", lambda: pascal)
        with pytest.raises(ValueError, match="choose the T4"):
            check_quantization(RunConfig(**MINIMAL, load_in_4bit=True))

    def test_4bit_config_on_turing_is_fine(self, monkeypatch):
        turing = Device("cuda", "Tesla T4", (7, 5), 15.8, supports_bf16=False)
        monkeypatch.setattr("docbench.train.loop.detect_device", lambda: turing)
        check_quantization(RunConfig(**MINIMAL, load_in_4bit=True))

    def test_non_4bit_config_runs_anywhere(self, monkeypatch):
        pascal = Device("cuda", "Tesla P100-PCIE-16GB", (6, 0), 16.0, supports_bf16=False)
        monkeypatch.setattr("docbench.train.loop.detect_device", lambda: pascal)
        check_quantization(RunConfig(**MINIMAL, load_in_4bit=False))

    def test_detection_degrades_without_torch(self):
        device = detect_device()
        assert device.kind in {"cuda", "mps", "cpu", "unknown"}
        assert device.describe()


class TestToken:
    def test_repr_does_not_leak(self):
        assert "hunter2" not in repr(Token("hunter2", "environment"))

    def test_str_does_not_leak(self):
        assert "hunter2" not in str(Token("hunter2", "environment"))

    def test_fstring_does_not_leak(self):
        assert "hunter2" not in f"{Token('hunter2', 'environment')}"

    def test_reveal_returns_the_value(self):
        assert Token("hunter2", "environment").reveal() == "hunter2"

    def test_reads_from_environment(self, monkeypatch):
        monkeypatch.setenv("DOCFIT_TEST_TOKEN", "abc123")
        token = get_token("DOCFIT_TEST_TOKEN")
        assert token is not None and token.reveal() == "abc123"
        assert token.source == "environment"

    def test_missing_required_token_explains_both_sources(self, monkeypatch):
        monkeypatch.delenv("DOCFIT_ABSENT", raising=False)
        with pytest.raises(MissingTokenError, match="Add-ons"):
            get_token("DOCFIT_ABSENT")

    def test_missing_optional_token_is_none(self, monkeypatch):
        monkeypatch.delenv("DOCFIT_ABSENT", raising=False)
        assert get_token("DOCFIT_ABSENT", required=False) is None
