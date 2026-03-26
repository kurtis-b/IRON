import pytest

from iron.applications.transformer_layer.gpu_power import (
    parse_rocm_smi_average_power_w,
)


def test_parse_rocm_smi_average_power_w_reads_card_payload():
    stdout = """
    {
      "card0": {
        "Average Graphics Package Power (W)": "47.0W"
      }
    }
    """

    assert parse_rocm_smi_average_power_w(stdout, card_label="card0") == 47.0


def test_parse_rocm_smi_average_power_w_returns_none_when_missing():
    stdout = '{"card0": {"Temperature (Sensor edge) (C)": "55.0"}}'

    assert parse_rocm_smi_average_power_w(stdout, card_label="card0") is None
