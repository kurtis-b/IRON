import json

from iron.applications.transformer_layer.operator_runlist_component_study import (
    load_component_study_config,
    run_component_study,
)


def test_load_component_study_config_resolves_relative_output_path(tmp_path):
    config_path = tmp_path / "component.json"
    config_path.write_text(
        json.dumps(
            {
                "study_cases": [
                    {
                        "case_id": "baseline_768",
                        "case_label": "baseline_768",
                        "layer_spec": {
                            "hidden_size": 768,
                            "intermediate_size": 3072,
                            "num_attention_heads": 12,
                            "batch_size": 1,
                            "seq_len": 128,
                            "dtype": "bfloat16",
                            "activation": "gelu",
                            "use_bias": False,
                            "layer_norm_eps": 1e-12,
                            "attention_mask_mode": "none",
                            "weights_source": "synthetic",
                            "source_model_name": None,
                            "source_layer_index": None,
                        },
                    }
                ],
                "seq_lens": [4096],
                "output_csv": "../results/components.csv",
            }
        ),
        encoding="utf-8",
    )

    loaded = load_component_study_config(config_path)

    assert loaded["output_csv"].endswith("/results/components.csv")


def test_run_component_study_tags_case_metadata_and_writes_csv(monkeypatch, tmp_path):
    output_csv = tmp_path / "component_rows.csv"

    def fake_run_component_check_isolated(**kwargs):
        spec = kwargs["spec"]
        assert kwargs["targets"] == "attn_scores"
        return [
            {
                "target": "attn_scores",
                "seq_len": spec.seq_len,
                "stable": True,
                "max_abs_diff": 0.0,
                "mean_abs_diff": 0.0,
            }
        ]

    monkeypatch.setattr(
        "iron.applications.transformer_layer.operator_runlist_component_study._run_component_check_isolated",
        fake_run_component_check_isolated,
    )

    rows = run_component_study(
        {
            "study_cases": [
                {
                    "case_id": "dense_4b_class",
                    "case_label": "dense_4b_class",
                    "layer_spec": {
                        "hidden_size": 2560,
                        "intermediate_size": 10240,
                        "num_attention_heads": 32,
                        "batch_size": 1,
                        "seq_len": 128,
                        "dtype": "bfloat16",
                        "activation": "gelu",
                        "use_bias": False,
                        "layer_norm_eps": 1e-12,
                        "attention_mask_mode": "none",
                        "weights_source": "synthetic",
                        "source_model_name": None,
                        "source_layer_index": None,
                    },
                }
            ],
            "seq_lens": [4096],
            "targets": "attn_scores",
            "repeats": 2,
            "seed": 7,
            "output_csv": str(output_csv),
        }
    )

    assert len(rows) == 1
    assert rows[0]["study_case_id"] == "dense_4b_class"
    assert rows[0]["hidden_size"] == 2560
    assert rows[0]["attention_head_size"] == 80
    assert output_csv.exists()
