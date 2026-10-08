import os
from dataclasses import dataclass

import torch
from safetensors.torch import load_file


RELEASED_LOSS_PREFIX = "loss_fns.KeyPhraseAlignmentLoss."
CURRENT_LOSS_PREFIX = "loss_fns.RadZeroLoss."


@dataclass
class CheckpointLoadResult:
    missing_keys: list
    unexpected_keys: list
    verification_key: str


def migrate_released_radzero_state_dict(state_dict):
    """Rename the released RadZero loss prefix to the current repository name."""
    return {
        (
            CURRENT_LOSS_PREFIX + key[len(RELEASED_LOSS_PREFIX) :]
            if key.startswith(RELEASED_LOSS_PREFIX)
            else key
        ): value
        for key, value in state_dict.items()
    }


def load_radzero_state_dict(model, checkpoint_path):
    """Load released RadZero weights while allowing explicitly new modules."""
    checkpoint_path = os.fspath(checkpoint_path)
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "model.safetensors")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    state_dict = migrate_released_radzero_state_dict(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)

    allowed_prefixes = getattr(model, "pretrained_missing_key_prefixes", ())
    unexpected_missing = [
        key for key in incompatible.missing_keys if not key.startswith(allowed_prefixes)
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Incompatible RadZero checkpoint: "
            f"missing={unexpected_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    verification_key = next(
        (
            key
            for key in sorted(state_dict)
            if not key.startswith(allowed_prefixes)
            and key.endswith(".weight")
            and state_dict[key].numel() > 1
        ),
        None,
    )
    if verification_key is None:
        raise RuntimeError("RadZero checkpoint contains no verifiable base tensor.")
    loaded_value = model.state_dict()[verification_key]
    expected_value = state_dict[verification_key].to(
        device=loaded_value.device, dtype=loaded_value.dtype
    )
    if not torch.equal(loaded_value, expected_value):
        raise RuntimeError(
            f"RadZero checkpoint verification failed for {verification_key}."
        )
    return CheckpointLoadResult(
        missing_keys=incompatible.missing_keys,
        unexpected_keys=incompatible.unexpected_keys,
        verification_key=verification_key,
    )


def load_selected_module_state_dict(model, checkpoint_path, prefixes):
    """Load only explicitly named module prefixes from a hierarchy checkpoint."""
    checkpoint_path = os.fspath(checkpoint_path)
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "model.safetensors")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = state_dict.get("state_dict", state_dict)
    selected = {
        key: value
        for key, value in state_dict.items()
        if any(key.startswith(prefix) for prefix in prefixes)
    }
    if not selected:
        raise RuntimeError(
            f"No checkpoint tensors matched prefixes {tuple(prefixes)}."
        )
    model_state = model.state_dict()
    unexpected = sorted(set(selected) - set(model_state))
    mismatched = sorted(
        key
        for key, value in selected.items()
        if key in model_state and value.shape != model_state[key].shape
    )
    if unexpected or mismatched:
        raise RuntimeError(
            f"Selected checkpoint is incompatible: unexpected={unexpected}, "
            f"shape_mismatch={mismatched}."
        )
    model.load_state_dict(selected, strict=False)
    return sorted(selected)
