#!/usr/bin/env python3
"""Host-CPU reference runtime for GR00T action head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


def _clone_shared_inputs(shared_inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.array(v, copy=True) for k, v in shared_inputs.items()}


def _load_model_cfg(model_dir: str | Path) -> dict[str, Any]:
    config_path = Path(model_dir).resolve() / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in model directory: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _load_action_head_state_dict(store: Any) -> dict[str, torch.Tensor]:
    prefix = "action_head."
    return {k[len(prefix) :]: store.get(k).to(torch.float32) for k in store.keys() if k.startswith(prefix)}


def run_full_action_head_reference(
    store: Any,
    args: argparse.Namespace,
    *,
    shared_inputs: dict[str, np.ndarray] | None,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    if shared_inputs is None:
        raise ValueError("run_full_action_head_reference requires shared_inputs")

    isaac_root = Path(args.isaac_gr00t_root).resolve()
    if not isaac_root.exists():
        raise FileNotFoundError(f"Isaac-GR00T root not found: {isaac_root}")
    if str(isaac_root) not in sys.path:
        sys.path.insert(0, str(isaac_root))

    from transformers.feature_extraction_utils import BatchFeature

    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead

    cfg_data = _load_model_cfg(args.model_dir)
    config = Gr00tN1d6Config(**cfg_data)
    config.action_horizon = args.action_horizon
    config.num_inference_timesteps = args.num_inference_timesteps

    action_head = Gr00tN1d6ActionHead(config).to(torch.float32).eval()
    state_dict = _load_action_head_state_dict(store)
    missing, unexpected = action_head.load_state_dict(state_dict, strict=False)
    missing_required = [k for k in missing if k != "mask_token"]
    if missing_required:
        raise RuntimeError(f"Missing required action-head keys: {missing_required[:10]}")
    if unexpected:
        raise RuntimeError(f"Unexpected action-head keys: {unexpected[:10]}")

    inputs = _clone_shared_inputs(shared_inputs)
    state = torch.from_numpy(inputs["state"]).to(torch.float32)
    backbone_features = torch.from_numpy(inputs["vl_embeds"]).to(torch.float32)
    image_mask = torch.from_numpy(inputs["image_mask"]).to(torch.bool)
    backbone_attention_mask = torch.from_numpy(inputs["backbone_attention_mask"]).to(torch.bool)
    embodiment_id = torch.from_numpy(inputs["embodiment_id"]).to(torch.long)
    actions = torch.from_numpy(inputs["actions"]).to(torch.float32)
    batch = int(state.shape[0])
    action_horizon = int(args.action_horizon)
    num_steps = int(args.num_inference_timesteps)

    backbone_output = BatchFeature(
        data={
            "backbone_features": backbone_features,
            "backbone_attention_mask": backbone_attention_mask,
            "image_mask": image_mask,
        }
    )
    action_input = BatchFeature(
        data={
            "state": state,
            "embodiment_id": embodiment_id,
        }
    )

    debug: dict[str, np.ndarray] = {}
    debug["input/state"] = inputs["state"].astype(np.float32)
    debug["input/vl_embeds"] = inputs["vl_embeds"].astype(np.float32)
    debug["input/image_mask"] = inputs["image_mask"]
    debug["input/backbone_attention_mask"] = inputs["backbone_attention_mask"]
    debug["input/actions_init"] = inputs["actions"].astype(np.float32)
    debug["input/embodiment_id"] = inputs["embodiment_id"]

    with torch.no_grad():
        backbone_output = action_head.process_backbone_output(backbone_output)
        vl_embeds = backbone_output.backbone_features
        state_features = action_head.state_encoder(action_input.state, embodiment_id)
        debug["encoded/vl_embeds"] = vl_embeds.detach().cpu().to(torch.float32).numpy()
        debug["encoded/state_features"] = state_features.detach().cpu().to(torch.float32).numpy()

        dt = 1.0 / float(num_steps)
        for t in range(num_steps):
            t_cont = float(t) / float(num_steps)
            t_discretized = int(t_cont * int(action_head.num_timestep_buckets))
            timesteps_tensor = torch.full(size=(batch,), fill_value=t_discretized, dtype=torch.long)

            action_features = action_head.action_encoder(actions, timesteps_tensor, embodiment_id)
            if config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long)
                pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs
            sa_embs = torch.cat((state_features, action_features), dim=1)

            if config.use_alternate_vl_dit:
                model_output, all_hidden_states = action_head.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    return_all_hidden_states=True,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output, all_hidden_states = action_head.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    return_all_hidden_states=True,
                )
            pred = action_head.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -action_horizon:, :]
            actions = actions + dt * pred_velocity

            step_prefix = f"step_{t:02d}"
            debug[f"{step_prefix}/timesteps"] = timesteps_tensor.detach().cpu().numpy()
            debug[f"{step_prefix}/action_features"] = (
                action_features.detach().cpu().to(torch.float32).numpy()
            )
            debug[f"{step_prefix}/sa_embs"] = sa_embs.detach().cpu().to(torch.float32).numpy()
            debug[f"{step_prefix}/model_output"] = (
                model_output.detach().cpu().to(torch.float32).numpy()
            )
            debug[f"{step_prefix}/pred_velocity"] = (
                pred_velocity.detach().cpu().to(torch.float32).numpy()
            )
            debug[f"{step_prefix}/actions"] = actions.detach().cpu().to(torch.float32).numpy()
            for i, hs in enumerate(all_hidden_states):
                debug[f"{step_prefix}/block_hidden_{i:02d}"] = (
                    hs.detach().cpu().to(torch.float32).numpy()
                )

    action_np = actions.detach().cpu().to(torch.float32).numpy()
    debug["output/actions"] = action_np
    if return_debug:
        return action_np, debug
    return action_np
