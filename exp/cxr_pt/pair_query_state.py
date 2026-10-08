import hashlib
import json
from pathlib import Path

import torch


STATE_SOURCE = "anatomy_head.vlcabs.evidence_tokens"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def save_static_anatomy_state(cache_path, state, metadata):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if state.dim() != 2 or not torch.isfinite(state).all():
        raise ValueError("Static anatomy state must be a finite [A, D] tensor.")
    metadata = dict(metadata)
    metadata.update(
        state_source=STATE_SOURCE,
        shape=list(state.shape),
        dtype="float32",
    )
    torch.save({"state": state.detach().float().cpu()}, cache_path)
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata_path


def load_static_anatomy_state(cache_path, expected_metadata=None):
    cache_path = Path(cache_path)
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if not cache_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Static anatomy cache is incomplete: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    state = payload["state"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("state_source") != STATE_SOURCE:
        raise ValueError("Static anatomy cache has a different state source.")
    if list(state.shape) != metadata.get("shape") or not torch.isfinite(state).all():
        raise ValueError("Static anatomy cache tensor does not match its metadata.")
    for key, expected in (expected_metadata or {}).items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Static anatomy cache provenance mismatch for {key}: "
                f"expected {expected!r}, found {metadata.get(key)!r}."
            )
    return state.float(), metadata


def anatomy_state_diagnostics(states, static_mean, left_right_pairs=None):
    states = states.detach().double().cpu()
    static_mean = static_mean.detach().double().cpu()
    if states.dim() != 3 or states.shape[1:] != static_mean.shape:
        raise ValueError("states must be [N, A, D] and match static_mean [A, D].")
    centered_norm = torch.linalg.vector_norm(states - static_mean, dim=-1)
    cosine_to_mean = torch.nn.functional.cosine_similarity(
        states, static_mean.unsqueeze(0), dim=-1
    )
    normalized = torch.nn.functional.normalize(states, dim=-1)
    cross_anatomy = torch.matmul(normalized, normalized.transpose(1, 2))
    off_diagonal = ~torch.eye(states.size(1), dtype=torch.bool)
    result = {
        "sample_count": int(states.size(0)),
        "per_anatomy_feature_mean": states.mean(dim=0).float().tolist(),
        "per_anatomy_feature_std": states.std(dim=0, unbiased=False).float().tolist(),
        "distance_to_mean": _distribution(centered_norm),
        "cosine_to_mean": _distribution(cosine_to_mean),
        "same_image_cross_anatomy_cosine": _distribution(
            cross_anatomy[:, off_diagonal]
        ),
        "left_right": [],
    }
    for left, right in left_right_pairs or []:
        similarities = torch.nn.functional.cosine_similarity(
            states[:, left], states[:, right], dim=-1
        )
        result["left_right"].append(
            {"left_index": left, "right_index": right, **_distribution(similarities)}
        )
    return result


def _distribution(values):
    values = values.flatten().float()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "p05": float(torch.quantile(values, 0.05).item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
    }
