# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Op-level unit tests for GR00T N1.6 ttnn kernels.

Shapes are extracted from gr00t_dryrun.log [op-perf] lines and cover all
three sub-models: SigLIP2 vision encoder, MLP1 connector, Qwen3 decoder,
and the AlternateVLDiT action head.

Each test case runs the ttnn op against a torch reference and checks PCC.
Pass threshold is 0.99 for element-wise ops, 0.98 for matmul/linear.

--sim mode
----------
Pass --sim to time each op under the gem5 simulator (see conftest.py).
Results are written to gr00t_ops_sim_baseline.csv.

Run:
    cd /workspaces/tensix/tt-metal
    pytest models/experimental/isaac_gr00t/tests/test_gr00t_ops.py -v
    pytest models/experimental/isaac_gr00t/tests/test_gr00t_ops.py -v --sim
"""

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PCC_MATMUL = 0.98
PCC_ELTWISE = 0.99

def _rand(*shape, dtype=torch.bfloat16):
    return torch.randn(shape, dtype=dtype)

def _upload(t, device):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

def _download(t):
    return ttnn.to_torch(t)


# ---------------------------------------------------------------------------
# matmul / linear  (2-D weight, 3-D/2-D activation)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("act_shape,w_shape", [
    # ── SigLIP2 ──────────────────────────────────────────────────────────────
    # patch embedding (2D: 576 patches × 588 pixel values)
    ((576, 588),        (588, 1152)),
    # attention Q/K/V/O projections
    ((1, 576, 1152),    (1152, 1152)),
    # FFN fc1 (fused linear_gelu) and fc2
    ((1, 576, 1152),    (1152, 4304)),
    ((1, 576, 4304),    (4304, 1152)),
    # ── MLP1 connector ───────────────────────────────────────────────────────
    # fc1 after pixel-shuffle (4608 = 2×2×1152)
    ((1, 144, 4608),    (4608, 2048)),
    # fc2
    ((1, 144, 2048),    (2048, 2048)),
    # ── Qwen3 decoder ────────────────────────────────────────────────────────
    # Q and O projections (full hidden → full hidden)
    ((1, 171, 2048),    (2048, 2048)),
    # K/V projections (GQA: 8 KV heads × 128 = 1024)
    ((1, 171, 2048),    (2048, 1024)),
    # FFN gate/up
    ((1, 171, 2048),    (2048, 6144)),
    # FFN down
    ((1, 171, 6144),    (6144, 2048)),
    # ── Action head — state encoder (2D after reshape_in) ────────────────────
    ((1, 128),          (128, 1024)),       # state_l1
    ((1, 1024),         (1024, 1536)),      # state_l2
    # ── Action head — time embedding ─────────────────────────────────────────
    ((1, 256),          (256, 1536)),       # time_l1  (256-dim sinusoidal → 1536)
    ((1, 1536),         (1536, 1536)),      # time_l2
    # ── Action head — action token widening (2D: 16 action steps) ────────────
    ((16, 128),         (128, 1536)),       # act_w1
    ((16, 3072),        (3072, 1536)),      # act_w2  (after concat with tau)
    ((16, 1536),        (1536, 1536)),      # act_w3
    # ── Action head — DiT block cross-attention K/V (2D) ─────────────────────
    ((171, 2048),       (2048, 1536)),      # block k/v linear (VL embeds → 1536)
    # ── Action head — DiT block self-attention Q/K/V/O  (2D) ──────────────────
    ((17, 1536),        (1536, 1536)),      # block q/k/v/o
    # ── Action head — DiT block FFN (2D) ─────────────────────────────────────
    ((17, 1536),        (1536, 6144)),      # ff_in
    ((17, 6144),        (6144, 1536)),      # ff_out
    # ── Action head — AdaLN norm linear ──────────────────────────────────────
    ((1, 1536),         (1536, 3072)),      # norm1_linear (→ scale+shift pair)
    # ── Action head — proj_out / decoder (2D) ────────────────────────────────
    ((17, 1536),        (1536, 1024)),      # proj_out_2
    ((17, 1024),        (1024, 1024)),      # dec_l1
    ((17, 1024),        (1024, 128)),       # dec_l2  (→ action dim)
])
def test_linear(act_shape, w_shape, device, sim_timer):
    a = _rand(*act_shape)
    w = _rand(*w_shape)
    ref = torch.matmul(a, w)

    tt_a = _upload(a, device)
    tt_w = _upload(w, device)
    out = sim_timer("matmul",
                    lambda: _download(ttnn.matmul(tt_a, tt_w)),
                    in_shapes=str([act_shape, w_shape]))
    assert_with_pcc(ref, out, PCC_MATMUL)


# ---------------------------------------------------------------------------
# Batched matmul — attention QK and PV
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a_shape,b_shape", [
    # SigLIP2 QK  (1, 16, 576, 72) × (1, 16, 72, 576)
    ((1, 16, 576, 72),  (1, 16, 72, 576)),
    # SigLIP2 PV  (1, 16, 576, 576) × (1, 16, 576, 72)
    ((1, 16, 576, 576), (1, 16, 576, 72)),
    # Qwen3 QK/PV — internal to ttnn.transformer.scaled_dot_product_attention,
    # but kernel coverage is still useful
    ((1, 16, 171, 128), (1, 16, 128, 171)),
    ((1, 16, 171, 171), (1, 16, 171, 128)),
    # Action head cross-attn QK (1, 32, 17, 48) × (1, 32, 48, 171)
    ((1, 32, 17, 48),   (1, 32, 48, 171)),
    # Action head cross-attn PV (1, 32, 17, 171) × (1, 32, 171, 48)
    ((1, 32, 17, 171),  (1, 32, 171, 48)),
    # Action head self-attn QK (1, 32, 17, 48) × (1, 32, 48, 17)
    ((1, 32, 17, 48),   (1, 32, 48, 17)),
    # Action head self-attn PV (1, 32, 17, 17) × (1, 32, 17, 48)
    ((1, 32, 17, 17),   (1, 32, 17, 48)),
])
def test_batched_matmul(a_shape, b_shape, device, sim_timer):
    a = _rand(*a_shape)
    b = _rand(*b_shape)
    ref = torch.matmul(a, b)

    tt_a = _upload(a, device)
    tt_b = _upload(b, device)
    out = sim_timer("matmul",
                    lambda: _download(ttnn.matmul(tt_a, tt_b)),
                    in_shapes=str([a_shape, b_shape]))
    assert_with_pcc(ref, out, PCC_MATMUL)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    (1, 16, 576, 576),   # SigLIP2 attention scores
    # Qwen3 attention scores are internal to ttnn.transformer.scaled_dot_product_attention
    (1, 32, 17, 17),     # action head self-attn
    (1, 32, 17, 171),    # action head cross-attn
])
def test_softmax(shape, device, sim_timer):
    a = _rand(*shape)
    ref = torch.softmax(a, dim=-1)

    tt_a = _upload(a, device)
    out = sim_timer("softmax",
                    lambda: _download(ttnn.softmax(tt_a, dim=-1)),
                    in_shapes=str([shape]))
    assert_with_pcc(ref, out, PCC_ELTWISE)


# ---------------------------------------------------------------------------
# Layer norm / RMS norm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape,normalized_shape", [
    # SigLIP2 pre-attn / pre-FFN layer norm
    ((1, 576, 1152),  (1152,)),
    # MLP1 connector layer norm (before fc1)
    ((1, 144, 4608),  (4608,)),
    # Qwen3 RMS norm
    ((1, 171, 2048),  (2048,)),
    # Action head DiT block ln1/ln2
    ((1, 17, 1536),   (1536,)),
])
def test_layer_norm(shape, normalized_shape, device, sim_timer):
    a = _rand(*shape)
    w = torch.ones(normalized_shape, dtype=torch.bfloat16)
    b = torch.zeros(normalized_shape, dtype=torch.bfloat16)
    ref = torch.nn.functional.layer_norm(a, normalized_shape, weight=w, bias=b)

    tt_a = _upload(a, device)
    tt_w = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    tt_b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    out = sim_timer("layer_norm",
                    lambda: _download(ttnn.layer_norm(tt_a, weight=tt_w, bias=tt_b)),
                    in_shapes=str([shape, normalized_shape]))
    assert_with_pcc(ref, out, PCC_ELTWISE)


# ---------------------------------------------------------------------------
# Element-wise: add
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    (1, 576, 1152),   # SigLIP2 residual / pos embed add
    (1, 171, 2048),   # Qwen3 residual
    (1, 17, 1536),    # action head block residual
    (1, 16, 1536),    # action head add_pos
    (1, 1536),        # action head scalar add (time/state)
])
def test_add(shape, device, sim_timer):
    a = _rand(*shape)
    b = _rand(*shape)
    ref = a + b

    tt_a = _upload(a, device)
    tt_b = _upload(b, device)
    out = sim_timer("add",
                    lambda: _download(ttnn.add(tt_a, tt_b)),
                    in_shapes=str([shape, shape]))
    assert_with_pcc(ref, out, PCC_ELTWISE)


# ---------------------------------------------------------------------------
# Element-wise: multiply
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    (1, 171, 6144),   # Qwen3 FFN gate×up
    (1, 17, 1536),    # action head ada_mul / proj_out_affine_mul
    (1, 16, 1536),    # action head gate×up (act_w2 output)
    (1, 1536),        # action head scale_plus_one
    (1, 16, 128),     # action head step delta_scale
])
def test_multiply(shape, device, sim_timer):
    a = _rand(*shape)
    b = _rand(*shape)
    ref = a * b

    tt_a = _upload(a, device)
    tt_b = _upload(b, device)
    out = sim_timer("multiply",
                    lambda: _download(ttnn.multiply(tt_a, tt_b)),
                    in_shapes=str([shape, shape]))
    assert_with_pcc(ref, out, PCC_ELTWISE)


# ---------------------------------------------------------------------------
# Activation functions: silu, gelu, relu
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    (1, 171, 6144),   # Qwen3 FFN gate silu
    (1, 16, 1536),    # action head action.silu
    (1, 1536),        # action head time_silu / ada_silu
    (1, 3072),        # action head proj_out_1_silu (after norm1_linear)
])
def test_silu(shape, device, sim_timer):
    a = _rand(*shape)
    ref = torch.nn.functional.silu(a)

    tt_a = _upload(a, device)
    out = sim_timer("silu",
                    lambda: _download(ttnn.silu(tt_a)),
                    in_shapes=str([shape]))
    assert_with_pcc(ref, out, PCC_ELTWISE)


@pytest.mark.parametrize("shape", [
    (1, 576, 4304),   # SigLIP2 FFN fc1 gelu output
    (1, 144, 2048),   # MLP1 fc1 gelu output
    (1, 17, 6144),    # action head block ff_gelu
])
def test_gelu(shape, device, sim_timer):
    a = _rand(*shape)
    ref = torch.nn.functional.gelu(a)

    tt_a = _upload(a, device)
    out = sim_timer("gelu",
                    lambda: _download(ttnn.gelu(tt_a)),
                    in_shapes=str([shape]))
    assert_with_pcc(ref, out, PCC_ELTWISE)


@pytest.mark.parametrize("shape", [
    (1, 1, 1024),    # state_relu (state encoder hidden)
    (1, 17, 1024),   # dec_relu (action decoder hidden, 17 steps)
])
def test_relu(shape, device, sim_timer):
    a = _rand(*shape)
    ref = torch.nn.functional.relu(a)

    tt_a = _upload(a, device)
    out = sim_timer("relu",
                    lambda: _download(ttnn.relu(tt_a)),
                    in_shapes=str([shape]))
    assert_with_pcc(ref, out, PCC_ELTWISE)


# ---------------------------------------------------------------------------
# Permute / transpose
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape,perm", [
    # SigLIP2: (1,576,16,72) -> (1,16,576,72)
    ((1, 576, 16, 72),  (0, 2, 1, 3)),
    # SigLIP2: (1,16,576,72) -> (1,576,16,72)
    ((1, 16, 576, 72),  (0, 2, 1, 3)),
    # SigLIP2 K transpose: (1,16,576,72) -> (1,16,72,576)
    ((1, 16, 576, 72),  (0, 1, 3, 2)),
    # Qwen3 (CPU-side, tested here for TTNN kernel coverage):
    # (1,171,16,128) -> (1,16,171,128)
    ((1, 171, 16, 128), (0, 2, 1, 3)),
    # Qwen3 K transpose: (1,16,171,128) -> (1,16,128,171)
    ((1, 16, 171, 128), (0, 1, 3, 2)),
    # Action head Q: (1,17,32,48) -> (1,32,17,48)
    ((1, 17, 32, 48),   (0, 2, 1, 3)),
    # Action head K (self-attn): (1,17,32,48) -> (1,32,48,17)
    ((1, 17, 32, 48),   (0, 2, 3, 1)),
    # Action head ctx: (1,32,17,48) -> (1,17,32,48)
    ((1, 32, 17, 48),   (0, 2, 1, 3)),
    # Action head cross-attn K: (1,171,32,48) -> (1,32,48,171)
    ((1, 171, 32, 48),  (0, 2, 3, 1)),
    # Action head cross-attn V: (1,171,32,48) -> (1,32,171,48)
    ((1, 171, 32, 48),  (0, 2, 1, 3)),
])
def test_permute(shape, perm, device, sim_timer):
    a = _rand(*shape)
    ref = a.permute(perm).contiguous()

    tt_a = _upload(a, device)
    out = sim_timer("permute",
                    lambda: _download(ttnn.permute(tt_a, perm)),
                    in_shapes=str([shape]))
    assert_with_pcc(ref, out, PCC_ELTWISE)


# ---------------------------------------------------------------------------
# Reshape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_shape,out_shape", [
    # ── SigLIP2 ──────────────────────────────────────────────────────────────
    ((576, 1152),           (1, 576, 1152)),
    ((1, 576, 1152),        (1, 576, 16, 72)),
    ((1, 576, 16, 72),      (1, 576, 1152)),
    # ── MLP1 connector (pixel-shuffle) ───────────────────────────────────────
    ((1, 576, 1152),        (1, 144, 4608)),
    # ── Qwen3 ────────────────────────────────────────────────────────────────
    # concat_heads output: (B, 1, T, H*D) → (B, T, H*D)
    ((1, 1, 171, 2048),     (1, 171, 2048)),
    # ── Action head — state encoder ──────────────────────────────────────────
    ((1, 1, 128),           (1, 128)),
    ((1, 1024),             (1, 1, 1024)),
    ((1, 1, 1024),          (1, 1024)),
    ((1, 1536),             (1, 1, 1536)),
    # ── Action head — action widening ────────────────────────────────────────
    ((1, 16, 128),          (16, 128)),
    ((16, 1536),            (1, 16, 1536)),
    ((1, 16, 1536),         (16, 1536)),
    ((1, 16, 3072),         (16, 3072)),
    # ── Action head — DiT block Q/K/V/O ──────────────────────────────────────
    ((1, 17, 1536),         (17, 1536)),
    ((17, 1536),            (1, 17, 1536)),
    # ── Action head — DiT block cross-attn K/V ───────────────────────────────
    ((1, 171, 2048),        (171, 2048)),
    ((171, 1536),           (1, 171, 1536)),
    # ── Action head — DiT block attention head split/merge ───────────────────
    ((1, 17, 1536),         (1, 17, 32, 48)),
    ((1, 17, 32, 48),       (1, 17, 1536)),
    # ── Action head — DiT block FFN ──────────────────────────────────────────
    ((1, 17, 6144),         (17, 6144)),
    ((17, 6144),            (1, 17, 6144)),
    # ── Action head — proj_out / decoder ─────────────────────────────────────
    ((17, 1024),            (1, 17, 1024)),
    ((1, 17, 1024),         (17, 1024)),
    ((17, 128),             (1, 17, 128)),
])
def test_reshape(in_shape, out_shape, device, sim_timer):
    a = _rand(*in_shape)
    ref = a.reshape(out_shape)

    tt_a = _upload(a, device)
    out = sim_timer("reshape",
                    lambda: _download(ttnn.reshape(tt_a, out_shape)),
                    in_shapes=str([in_shape]))
    assert_with_pcc(ref, out, PCC_ELTWISE)


# ---------------------------------------------------------------------------
# Concat
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shapes,dim", [
    # Action head: concat VL embeds + action tokens along seq dim
    ([(1, 144, 2048), (1, 17, 2048)], 1),
    ([(1, 576, 1152), (1, 17, 1152)], 1),
    # concat along last dim
    ([(1, 17, 1024), (1, 17, 512)], -1),
])
def test_concat(shapes, dim, device, sim_timer):
    tensors = [_rand(*s) for s in shapes]
    ref = torch.cat(tensors, dim=dim)

    tt_tensors = [_upload(t, device) for t in tensors]
    out = sim_timer("concat",
                    lambda: _download(ttnn.concat(tt_tensors, dim)),
                    in_shapes=str(shapes))
    assert_with_pcc(ref, out, PCC_ELTWISE)


# ---------------------------------------------------------------------------
# Repeat / broadcast
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape,repeats,dim", [
    # Action head cross-attn: expand mask (1,1,1,171) -> (1,32,17,171)
    ((1, 1, 1, 171), [1, 32, 17, 1], None),
])
def test_repeat(shape, repeats, dim, device, sim_timer):
    a = _rand(*shape)
    ref = a.expand(repeats) if dim is None else a.repeat_interleave(repeats, dim=dim)

    tt_a = _upload(a, device)
    out = sim_timer("repeat",
                    lambda: _download(ttnn.repeat(tt_a, ttnn.Shape(repeats))),
                    in_shapes=str([shape]))
    assert_with_pcc(ref, out, PCC_ELTWISE)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape,chunks,dim", [
    # Action head ada_split and proj_out_1_split: (1,3072) -> 2 × (1,1536)
    ((1, 3072),    2, -1),
])
def test_split(shape, chunks, dim, device, sim_timer):
    a = _rand(*shape)
    ref_parts = torch.chunk(a, chunks, dim=dim)

    tt_a = _upload(a, device)
    # split returns a list; download all parts inside the timer so END marks after sync
    parts = sim_timer("split",
                      lambda: [_download(p) for p in ttnn.split(tt_a, a.shape[dim] // chunks, dim)],
                      in_shapes=str([shape]))

    for ref, got in zip(ref_parts, parts):
        assert_with_pcc(ref, got, PCC_ELTWISE)
