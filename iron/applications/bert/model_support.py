#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from pathlib import Path

SUPPORTED_MODEL_FAMILIES = ("bert", "roberta", "distilbert")
APP_DIR = Path(__file__).resolve().parent
DEFAULT_STUDY_MANIFEST = APP_DIR / "study" / "encoder_only_models.json"
DEFAULT_MODELS_DIR = APP_DIR / "models"
ROOT_PREFIX_BY_FAMILY = {
    "bert": "bert",
    "roberta": "roberta",
    "distilbert": "distilbert",
}
DISPLAY_NAME_BY_FAMILY = {
    "bert": "BertModel(add_pooling_layer=False)",
    "roberta": "RobertaModel(add_pooling_layer=False)",
    "distilbert": "DistilBertModel",
}
MODEL_FILENAME = "model.safetensors"
HF_CONFIG_FILENAME = "config.json"


def _model_config_to_dict(model_config):
    if isinstance(model_config, dict):
        return dict(model_config)
    if hasattr(model_config, "__dict__"):
        return dict(vars(model_config))
    raise TypeError(f"Unsupported model_config type: {type(model_config)!r}")


def normalize_model_family(model_type):
    if model_type is None:
        return "bert"

    normalized = str(model_type).strip().lower().replace("_", "-")
    aliases = {
        "bert": "bert",
        "roberta": "roberta",
        "distilbert": "distilbert",
        "distil-bert": "distilbert",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported model_type {model_type!r}. Supported: {SUPPORTED_MODEL_FAMILIES}"
        )
    return aliases[normalized]


def canonicalize_model_config(model_config):
    cfg = _model_config_to_dict(model_config)
    family = normalize_model_family(cfg.get("model_type", "bert"))
    cfg["model_type"] = family

    if family == "distilbert":
        cfg.setdefault("hidden_size", cfg.get("dim"))
        cfg.setdefault("num_attention_heads", cfg.get("n_heads"))
        cfg.setdefault("num_hidden_layers", cfg.get("n_layers"))
        cfg.setdefault("intermediate_size", cfg.get("hidden_dim"))
        cfg.setdefault("hidden_dropout_prob", cfg.get("dropout", 0.1))
        cfg.setdefault(
            "attention_probs_dropout_prob", cfg.get("attention_dropout", 0.1)
        )
        cfg.setdefault("hidden_act", cfg.get("activation", "gelu"))
        cfg.setdefault("type_vocab_size", 0)
    elif family == "roberta":
        cfg["type_vocab_size"] = 1
    else:
        cfg.setdefault("type_vocab_size", 2)

    required = (
        "vocab_size",
        "hidden_size",
        "num_attention_heads",
        "num_hidden_layers",
        "intermediate_size",
        "max_position_embeddings",
        "pad_token_id",
        "hidden_dropout_prob",
        "attention_probs_dropout_prob",
    )
    missing = [name for name in required if cfg.get(name) is None]
    if missing:
        raise ValueError(f"Missing required model_config fields: {missing}")
    return cfg


def canonicalize_app_config_dict(config_json, seq_len=None):
    canonical = deepcopy(config_json)
    if "model_config" not in canonical:
        raise ValueError("Config JSON does not contain model_config")
    canonical["model_config"] = canonicalize_model_config(canonical["model_config"])
    if seq_len is not None:
        canonical["model_config"]["max_position_embeddings"] = (
            runtime_max_position_embeddings(canonical["model_config"], seq_len)
        )
    return canonical


def model_uses_token_type_ids(model_config):
    family = normalize_model_family(
        _model_config_to_dict(model_config).get("model_type")
    )
    return family != "distilbert"


def display_name_for_model(model_config):
    family = normalize_model_family(
        _model_config_to_dict(model_config).get("model_type")
    )
    return DISPLAY_NAME_BY_FAMILY[family]


def runtime_max_position_embeddings(model_config, seq_len):
    cfg = canonicalize_model_config(model_config)
    seq_len = int(seq_len)
    if cfg["model_type"] == "roberta":
        return seq_len + int(cfg["pad_token_id"]) + 1
    return seq_len


def estimate_encoder_forward_flops(model_config, seq_len, batch_size=1):
    cfg = canonicalize_model_config(model_config)
    seq_len = int(seq_len)
    batch_size = int(batch_size)
    hidden_size = int(cfg["hidden_size"])
    intermediate_size = int(cfg["intermediate_size"])
    num_hidden_layers = int(cfg["num_hidden_layers"])

    # Dominant forward-pass GEMM estimate for encoder-only transformers.
    # Multiply-add pairs count as 2 FLOPs.
    projection_flops = 8 * batch_size * seq_len * hidden_size * hidden_size
    attention_matmul_flops = 4 * batch_size * seq_len * seq_len * hidden_size
    ffn_flops = 4 * batch_size * seq_len * hidden_size * intermediate_size
    return num_hidden_layers * (projection_flops + attention_matmul_flops + ffn_flops)


def build_hf_encoder_model(model_config):
    cfg = canonicalize_model_config(model_config)
    family = cfg["model_type"]

    if family == "bert":
        from transformers import BertConfig, BertModel

        hf_config = BertConfig.from_dict(cfg)
        model = BertModel(hf_config, add_pooling_layer=False)
        return family, hf_config, model

    if family == "roberta":
        from transformers import RobertaConfig, RobertaModel

        hf_config = RobertaConfig.from_dict(cfg)
        model = RobertaModel(hf_config, add_pooling_layer=False)
        return family, hf_config, model

    from transformers import DistilBertConfig, DistilBertModel

    hf_cfg = {
        "model_type": "distilbert",
        "vocab_size": int(cfg["vocab_size"]),
        "max_position_embeddings": int(cfg["max_position_embeddings"]),
        "n_layers": int(cfg["num_hidden_layers"]),
        "n_heads": int(cfg["num_attention_heads"]),
        "dim": int(cfg["hidden_size"]),
        "hidden_dim": int(cfg["intermediate_size"]),
        "dropout": float(cfg["hidden_dropout_prob"]),
        "attention_dropout": float(cfg["attention_probs_dropout_prob"]),
        "activation": cfg.get("hidden_act", "gelu"),
        "pad_token_id": int(cfg["pad_token_id"]),
        "sinusoidal_pos_embds": bool(cfg.get("sinusoidal_pos_embds", False)),
        "qa_dropout": float(cfg.get("qa_dropout", 0.1)),
        "seq_classif_dropout": float(cfg.get("seq_classif_dropout", 0.2)),
    }
    if cfg.get("layer_norm_eps") is not None:
        hf_cfg["layer_norm_eps"] = float(cfg["layer_norm_eps"])
    hf_config = DistilBertConfig.from_dict(hf_cfg)
    model = DistilBertModel(hf_config)
    return family, hf_config, model


def normalize_layernorm_key(key):
    key = key.replace("LayerNorm.gamma", "LayerNorm.weight")
    key = key.replace("LayerNorm.beta", "LayerNorm.bias")
    return key


def _is_backbone_key(family, key):
    if family in ("bert", "roberta"):
        return (
            key.startswith("embeddings.")
            or key.startswith("encoder.")
            or key.startswith("pooler.")
        )
    if family == "distilbert":
        return key.startswith("embeddings.") or key.startswith("transformer.")
    return False


def extract_backbone_state_dict(raw_weights, model_config):
    cfg = canonicalize_model_config(model_config)
    family = cfg["model_type"]
    prefix = f"{ROOT_PREFIX_BY_FAMILY[family]}."
    has_prefix = any(key.startswith(prefix) for key in raw_weights)

    stripped = {}
    for key, value in raw_weights.items():
        if has_prefix:
            if not key.startswith(prefix):
                continue
            candidate = key[len(prefix) :]
        else:
            candidate = key
            if not _is_backbone_key(family, candidate):
                continue

        candidate = normalize_layernorm_key(candidate)
        if candidate == "embeddings.position_ids":
            continue
        if family in ("bert", "roberta") and candidate.startswith("pooler."):
            continue
        stripped[candidate] = value
    return stripped


def extend_or_trim_position_embeddings(state_dict, seq_len):
    key = "embeddings.position_embeddings.weight"
    if key not in state_dict:
        return
    position_weights = state_dict[key]
    if position_weights.shape[0] == seq_len:
        return
    if position_weights.shape[0] > seq_len:
        state_dict[key] = position_weights[:seq_len, :]
        return
    repeats = (seq_len // position_weights.shape[0]) + 1
    state_dict[key] = position_weights.repeat(repeats, 1)[:seq_len, :]


def canonicalize_local_backbone_weights(raw_weights, model_config):
    cfg = canonicalize_model_config(model_config)
    family = cfg["model_type"]
    stripped = extract_backbone_state_dict(raw_weights, cfg)

    if family in ("bert", "roberta"):
        return stripped

    canonical = {
        "embeddings.word_embeddings.weight": stripped[
            "embeddings.word_embeddings.weight"
        ],
        "embeddings.position_embeddings.weight": stripped[
            "embeddings.position_embeddings.weight"
        ],
        "embeddings.LayerNorm.weight": stripped["embeddings.LayerNorm.weight"],
        "embeddings.LayerNorm.bias": stripped["embeddings.LayerNorm.bias"],
    }

    for layer_idx in range(int(cfg["num_hidden_layers"])):
        src = f"transformer.layer.{layer_idx}"
        dst = f"encoder.layer.{layer_idx}"
        canonical[f"{dst}.attention.self.query.weight"] = stripped[
            f"{src}.attention.q_lin.weight"
        ]
        canonical[f"{dst}.attention.self.query.bias"] = stripped[
            f"{src}.attention.q_lin.bias"
        ]
        canonical[f"{dst}.attention.self.key.weight"] = stripped[
            f"{src}.attention.k_lin.weight"
        ]
        canonical[f"{dst}.attention.self.key.bias"] = stripped[
            f"{src}.attention.k_lin.bias"
        ]
        canonical[f"{dst}.attention.self.value.weight"] = stripped[
            f"{src}.attention.v_lin.weight"
        ]
        canonical[f"{dst}.attention.self.value.bias"] = stripped[
            f"{src}.attention.v_lin.bias"
        ]
        canonical[f"{dst}.attention.output.dense.weight"] = stripped[
            f"{src}.attention.out_lin.weight"
        ]
        canonical[f"{dst}.attention.output.dense.bias"] = stripped[
            f"{src}.attention.out_lin.bias"
        ]
        canonical[f"{dst}.attention.output.LayerNorm.weight"] = stripped[
            f"{src}.sa_layer_norm.weight"
        ]
        canonical[f"{dst}.attention.output.LayerNorm.bias"] = stripped[
            f"{src}.sa_layer_norm.bias"
        ]
        canonical[f"{dst}.intermediate.dense.weight"] = stripped[
            f"{src}.ffn.lin1.weight"
        ]
        canonical[f"{dst}.intermediate.dense.bias"] = stripped[f"{src}.ffn.lin1.bias"]
        canonical[f"{dst}.output.dense.weight"] = stripped[f"{src}.ffn.lin2.weight"]
        canonical[f"{dst}.output.dense.bias"] = stripped[f"{src}.ffn.lin2.bias"]
        canonical[f"{dst}.output.LayerNorm.weight"] = stripped[
            f"{src}.output_layer_norm.weight"
        ]
        canonical[f"{dst}.output.LayerNorm.bias"] = stripped[
            f"{src}.output_layer_norm.bias"
        ]

    return canonical


def load_study_manifest(manifest_path=None):
    path = Path(manifest_path) if manifest_path is not None else DEFAULT_STUDY_MANIFEST
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    entries = manifest.get("study_models", [])
    if not isinstance(entries, list):
        raise ValueError(f"Malformed study manifest: {path}")
    return path.resolve(), entries


def get_study_entry(study_id, manifest_path=None):
    manifest_file, entries = load_study_manifest(manifest_path)
    for entry in entries:
        if entry.get("study_id") == study_id:
            resolved = dict(entry)
            resolved["_manifest_path"] = str(manifest_file)
            return resolved
    available = sorted(entry.get("study_id", "") for entry in entries)
    raise ValueError(f"Unknown study_id {study_id!r}. Available study ids: {available}")


def default_model_dir_for_study(study_id, models_root=None):
    root = Path(models_root) if models_root is not None else DEFAULT_MODELS_DIR
    return root.resolve() / study_id


def resolve_study_paths(study_id, manifest_path=None, models_root=None):
    entry = get_study_entry(study_id, manifest_path=manifest_path)
    manifest_file = Path(entry["_manifest_path"])
    app_dir = manifest_file.parent.parent
    model_dir = default_model_dir_for_study(study_id, models_root=models_root)
    config_file = (app_dir / entry["config_file"]).resolve()
    return {
        "study_id": study_id,
        "hf_model_id": entry["hf_model_id"],
        "family": entry["family"],
        "npu_ready": bool(entry.get("npu_ready", False)),
        "config_file_path": str(config_file),
        "model_dir": str(model_dir),
        "weights_file_path": str(model_dir / MODEL_FILENAME),
        "hf_config_path": str(model_dir / HF_CONFIG_FILENAME),
        "manifest_path": str(manifest_file),
    }
