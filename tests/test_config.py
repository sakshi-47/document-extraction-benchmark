"""Run configuration, hardware capability and token handling tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from docbench.auth import Token
from docbench.hardware import Device
from docbench.train.config import RunConfig, load_run_config
from docbench.train.loop import check_quantization, resolve_precision

MINIMAL = {"name": "t", "base_model": "some/model"}
P100 = Device("cuda", "Tesla P100-PCIE-16GB", (6, 0), 16.0, supports_bf16=False)
T4 = Device("cuda", "Tesla T4", (7, 5), 15.8, supports_bf16=False)


def test_typo_in_a_config_key_is_an_error(tmp_path):
    """Ignored silently, the run would not be the run the file describes."""
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(MINIMAL | {"learing_rate": 1e-4}))
    with pytest.raises(ValueError, match="learing_rate"):
        load_run_config(path)


def test_shipped_configs_are_valid():
    """CI would otherwise be advertising a run that cannot start."""
    for name in ("smoke.yaml", "qwen2vl_lora.yaml"):
        assert load_run_config(Path(__file__).parent.parent / "configs" / name).name


def test_explicit_bf16_on_pre_ampere_is_an_error(monkeypatch):
    """bf16 needs Ampere. Neither Kaggle card has it."""
    monkeypatch.setattr("docbench.train.loop.detect_device", lambda: P100)
    with pytest.raises(ValueError, match="pre-Ampere"):
        resolve_precision(RunConfig(**MINIMAL, precision="bfloat16"))


def test_pascal_cannot_run_4bit_but_turing_can():
    """bitsandbytes 4-bit needs compute 7.5+: T4 qualifies, P100 does not."""
    assert not P100.supports_int4
    assert T4.supports_int4
    assert T4.torch_dtype == "float16", "Turing has no bf16 either"


def test_4bit_config_on_pascal_fails_before_the_model_downloads(monkeypatch):
    """Seconds here versus a chunk of a weekly GPU allowance mid-run."""
    monkeypatch.setattr("docbench.train.loop.detect_device", lambda: P100)
    with pytest.raises(ValueError, match="choose the T4"):
        check_quantization(RunConfig(**MINIMAL, load_in_4bit=True))


def test_token_never_renders_its_value():
    """An .ipynb keeps cell outputs, so one print would persist forever."""
    token = Token("hunter2", "kaggle-secrets")
    assert "hunter2" not in repr(token)
    assert "hunter2" not in str(token)
    assert "hunter2" not in f"{token}"
    assert token.reveal() == "hunter2"
