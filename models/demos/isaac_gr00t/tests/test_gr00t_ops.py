# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.isaac_gr00t.demo.gr00t import (
    run_cpu_gr00t_action_head_reference,
    run_ttnn_decoder_projection,
)


def _make_reference_tensors(
    *, batch: int, action_horizon: int, action_dim: int, hidden_1536: int = 1536
) -> dict[str, np.ndarray]:
    state_in = 64
    rng = np.random.default_rng(0)

    tensors: dict[str, np.ndarray] = {
        "x": rng.standard_normal((batch, state_in), dtype=np.float32),
        "state_l1_w": rng.standard_normal((state_in, 1024), dtype=np.float32),
        "state_l1_b": rng.standard_normal((1024,), dtype=np.float32),
        "state_l2_w": rng.standard_normal((1024, hidden_1536), dtype=np.float32),
        "state_l2_b": rng.standard_normal((hidden_1536,), dtype=np.float32),
        "act_w1_w": rng.standard_normal((action_dim, hidden_1536), dtype=np.float32),
        "act_w1_b": rng.standard_normal((hidden_1536,), dtype=np.float32),
        "act_w2_w": rng.standard_normal((hidden_1536 * 2, hidden_1536), dtype=np.float32),
        "act_w2_b": rng.standard_normal((hidden_1536,), dtype=np.float32),
        "act_w3_w": rng.standard_normal((hidden_1536, hidden_1536), dtype=np.float32),
        "act_w3_b": rng.standard_normal((hidden_1536,), dtype=np.float32),
        "time_l1_w": rng.standard_normal((hidden_1536, 256), dtype=np.float32),
        "time_l1_b": rng.standard_normal((hidden_1536,), dtype=np.float32),
        "time_l2_w": rng.standard_normal((hidden_1536, hidden_1536), dtype=np.float32),
        "time_l2_b": rng.standard_normal((hidden_1536,), dtype=np.float32),
        "attn_q_w": rng.standard_normal((hidden_1536, hidden_1536), dtype=np.float32),
        "attn_q_b": rng.standard_normal((hidden_1536,), dtype=np.float32),
        "attn_k_w": rng.standard_normal((hidden_1536, 2048), dtype=np.float32),
        "attn_k_b": rng.standard_normal((hidden_1536,), dtype=np.float32),
        "attn_v_w": rng.standard_normal((hidden_1536, 2048), dtype=np.float32),
        "attn_v_b": rng.standard_normal((hidden_1536,), dtype=np.float32),
        "attn_o_w": rng.standard_normal((hidden_1536, hidden_1536), dtype=np.float32),
        "attn_o_b": rng.standard_normal((hidden_1536,), dtype=np.float32),
        "proj_out_2_w": rng.standard_normal((1024, hidden_1536), dtype=np.float32),
        "proj_out_2_b": rng.standard_normal((1024,), dtype=np.float32),
        "dec_l1_w": rng.standard_normal((1024, 1024), dtype=np.float32),
        "dec_l1_b": rng.standard_normal((1024,), dtype=np.float32),
        "dec_l2_w": rng.standard_normal((1024, action_dim), dtype=np.float32),
        "dec_l2_b": rng.standard_normal((action_dim,), dtype=np.float32),
    }
    return tensors


@pytest.mark.parametrize(
    "batch, action_horizon, num_inference_timestamps, vl_seq_len, action_dim",
    [
        (1, 8, 4, 16, 128),
        (2, 5, 3, 12, 128),
    ],
)
def test_cpu_gr00t_action_head_reference_shapes(
    batch, action_horizon, num_inference_timestamps, vl_seq_len, action_dim
):
    tensors = _make_reference_tensors(
        batch=batch, action_horizon=action_horizon, action_dim=action_dim
    )
    args = SimpleNamespace(
        batch=batch,
        action_horizon=action_horizon,
        num_inference_timesteps=num_inference_timestamps,
        vl_seq_len=vl_seq_len,
        seed=0,
    )

    actions, pred_features, pred_velocity, decoder_w, decoder_b = (
        run_cpu_gr00t_action_head_reference(tensors, args)
    )

    assert actions.shape == (batch, action_horizon, action_dim)
    assert pred_features.shape == (batch, action_horizon, 1024)
    assert pred_velocity.shape == (batch, action_horizon, action_dim)
    assert decoder_w.shape == (1024, action_dim)
    assert decoder_b.shape == (action_dim,)


@torch.no_grad()
@pytest.mark.parametrize(
    "batch, action_horizon, hidden, action_dim",
    [
        (1, 5, 1024, 128),   # exercises padding path (M = 5)
        (4, 8, 1024, 128),   # no padding path (M = 32)
    ],
)
def test_ttnn_decoder_projection_matches_torch(
    batch, action_horizon, hidden, action_dim, device
):
    rng = np.random.default_rng(1)
    pred_features = rng.standard_normal(
        (batch, action_horizon, hidden), dtype=np.float32
    )
    decoder_w = rng.standard_normal((hidden, action_dim), dtype=np.float32)
    decoder_b = rng.standard_normal((action_dim,), dtype=np.float32)

    tt_output = run_ttnn_decoder_projection(
        pred_features, decoder_w, decoder_b, device
    )

    torch_output = (
        torch.from_numpy(pred_features.reshape(batch * action_horizon, hidden))
        @ torch.from_numpy(decoder_w)
        + torch.from_numpy(decoder_b)
    ).reshape(batch, action_horizon, action_dim)

    passing, pcc_message = comp_pcc(torch_output, torch.from_numpy(tt_output), 0.999)
    logger.info(f"PCC: {pcc_message}")
    logger.info(comp_allclose(torch_output, torch.from_numpy(tt_output)))

    assert tt_output.shape == (batch, action_horizon, action_dim)
    assert passing, f"Decoder projection PCC check failed: {pcc_message}"

