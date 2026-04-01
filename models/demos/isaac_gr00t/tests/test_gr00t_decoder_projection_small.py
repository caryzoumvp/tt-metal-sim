# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.isaac_gr00t.demo.gr00t import run_ttnn_decoder_projection


@torch.no_grad()
@pytest.mark.parametrize(
    "batch, action_horizon",
    [
        (1, 1),
        (1, 2),
    ],
)
def test_ttnn_decoder_projection_small_repro(batch, action_horizon, device):
    hidden = 1024
    action_dim = 128
    rng = np.random.default_rng(7)

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
    assert passing, f"Small decoder projection PCC check failed: {pcc_message}"
