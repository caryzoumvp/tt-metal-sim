# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Op-level unit tests for GR00T N1.6 ttnn kernels.

Shapes are extracted from gr00t_dryrun.log [op-perf] lines and cover all
three sub-models: SigLIP2 vision encoder, MLP1 connector, Qwen3 decoder,
and the AlternateVLDiT action head.

Each test case runs the ttnn op against a torch reference and checks PCC.
Pass threshold is 0.99 for element-wise ops, 0.98 for matmul/linear.

Coverage summary
----------------
Test                  Op                  Shapes covered
----                  --                  --------------
test_linear           ttnn.matmul         17 2D/3D weight shapes from siglip2,
                                          mlp1, qwen3, action head
test_batched_matmul   ttnn.matmul         8 4D QK/PV attention shapes
test_softmax          ttnn.softmax        4 attention score shapes
test_layer_norm       ttnn.layer_norm     4 shapes (siglip2, mlp1, qwen3,
                                          action head)
test_add              ttnn.add            5 residual/ada shapes
test_multiply         ttnn.multiply       5 gate/scale shapes
test_silu             ttnn.silu           6 shapes
test_gelu             ttnn.gelu           3 FFN shapes
test_relu             ttnn.relu           2 shapes
test_permute          ttnn.permute        10 head-reshape/transpose permutations
test_reshape          ttnn.reshape        15 shapes
test_concat           ttnn.concat         3 concat patterns
test_repeat           ttnn.repeat         mask broadcast
test_split            ttnn.split          ada-split patterns

Run:
    cd /workspaces/tensix/tt-metal
    pytest models/experimental/isaac_gr00t/tests/test_gr00t_ops.py -v
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
    # SigLIP2 attention Q/K/V/O projections
    ((1, 576, 1152), (1152, 1152)),
    # SigLIP2 FFN gate (linear_gelu fused)
    ((1, 576, 1152), (1152, 4304)),
    # SigLIP2 FFN down
    ((1, 576, 4304), (4304, 1152)),
    # SigLIP2 patch embedding
    ((576, 588), (588, 1152)),
    # MLP1 connector fc1 (linear_gelu fused)
    ((1, 144, 2048), (2048, 2048)),  # fc2
    ((1, 144, 4608), (4608, 2048)),  # fc1 (after layer-norm reshape)
    # Qwen3 Q/K/V projections (seq=171)
    ((1, 171, 2048), (2048, 2048)),
    # Qwen3 O projection
    ((1, 171, 2048), (2048, 1024)),
    # Qwen3 FFN gate/up
    ((1, 171, 2048), (2048, 6144)),
    # Qwen3 FFN down
    ((1, 171, 6144), (6144, 2048)),
    # Action head: various small linears
    ((1, 1536), (1536, 1536)),
    ((1, 1536), (1536, 3072)),
    ((1, 1024), (1024, 1536)),
    ((1, 128), (128, 1024)),
    ((1, 17, 1536), (1536, 1536)),
    ((1, 17, 1536), (1536, 3072)),
    ((1, 17, 1536), (1536, 6144)),
])
def test_linear(act_shape, w_shape, device):
    a = _rand(*act_shape)
    w = _rand(*w_shape)
    ref = torch.matmul(a, w)

    tt_a = _upload(a, device)
    tt_w = _upload(w, device)
    tt_out = ttnn.matmul(tt_a, tt_w)
    out = _download(tt_out)

    assert_with_pcc(ref, out, PCC_MATMUL)


# ---------------------------------------------------------------------------
# Batched matmul — attention QK and PV
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a_shape,b_shape", [
    # SigLIP2 QK  (1, 16, 576, 72) × (1, 16, 72, 576)
    ((1, 16, 576, 72),  (1, 16, 72, 576)),
    # SigLIP2 PV  (1, 16, 576, 576) × (1, 16, 576, 72)
    ((1, 16, 576, 576), (1, 16, 576, 72)),
    # Qwen3 QK  (1, 16, 171, 128) × (1, 16, 128, 171)
    ((1, 16, 171, 128), (1, 16, 128, 171)),
    # Qwen3 PV  (1, 16, 171, 171) × (1, 16, 171, 128)
    ((1, 16, 171, 171), (1, 16, 171, 128)),
    # Action head cross-attn QK (1, 32, 17, 48) × (1, 32, 48, 17)
    ((1, 32, 17, 48),  (1, 32, 48, 17)),
    # Action head cross-attn QK (1, 32, 17, 48) × (1, 32, 48, 171)
    ((1, 32, 17, 48),  (1, 32, 48, 171)),
    # Action head PV  (1, 32, 17, 17) × (1, 32, 17, 48)
    ((1, 32, 17, 17),  (1, 32, 17, 48)),
    # Action head cross-attn PV  (1, 32, 17, 171) × (1, 32, 171, 48)
    ((1, 32, 17, 171), (1, 32, 171, 48)),
])
def test_batched_matmul(a_shape, b_shape, device):
    a = _rand(*a_shape)
    b = _rand(*b_shape)
    ref = torch.matmul(a, b)

    tt_a = _upload(a, device)
    tt_b = _upload(b, device)
    tt_out = ttnn.matmul(tt_a, tt_b)
    out = _download(tt_out)

    assert_with_pcc(ref, out, PCC_MATMUL)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    (1, 16, 576, 576),   # SigLIP2
    (1, 16, 171, 171),   # Qwen3
    (1, 32, 17, 17),     # action head self-attn
    (1, 32, 17, 171),    # action head cross-attn
])
def test_softmax(shape, device):
    a = _rand(*shape)
    ref = torch.softmax(a, dim=-1)

    tt_a = _upload(a, device)
    tt_out = ttnn.softmax(tt_a, dim=-1)
    out = _download(tt_out)

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
    # Action head AdaLN / out norms
    ((1, 17, 1536),   (1536,)),
])
def test_layer_norm(shape, normalized_shape, device):
    a = _rand(*shape)
    w = torch.ones(normalized_shape, dtype=torch.bfloat16)
    b = torch.zeros(normalized_shape, dtype=torch.bfloat16)
    ref = torch.nn.functional.layer_norm(a, normalized_shape, weight=w, bias=b)

    tt_a = _upload(a, device)
    tt_w = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    tt_b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    tt_out = ttnn.layer_norm(tt_a, weight=tt_w, bias=tt_b)
    out = _download(tt_out)

    assert_with_pcc(ref, out, PCC_ELTWISE)


# ---------------------------------------------------------------------------
# Element-wise: add
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    (1, 576, 1152),   # SigLIP2 residual / pos embed add
    (1, 171, 2048),   # Qwen3 residual
    (1, 17, 1536),    # action head residual
    (1, 16, 1536),    # action head ada-add
    (1, 1536),        # action head scalar add
])
def test_add(shape, device):
    a = _rand(*shape)
    b = _rand(*shape)
    ref = a + b

    tt_a = _upload(a, device)
    tt_b = _upload(b, device)
    tt_out = ttnn.add(tt_a, tt_b)
    out = _download(tt_out)

    assert_with_pcc(ref, out, PCC_ELTWISE)


# ---------------------------------------------------------------------------
# Element-wise: multiply
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    (1, 171, 6144),   # Qwen3 FFN gate×up
    (1, 17, 1536),    # action head ada-mul
    (1, 16, 1536),    # action head gate mul
    (1, 1536),        # action head scalar mul
    (1, 16, 128),     # action head delta scale
])
def test_multiply(shape, device):
    a = _rand(*shape)
    b = _rand(*shape)
    ref = a * b

    tt_a = _upload(a, device)
    tt_b = _upload(b, device)
    tt_out = ttnn.multiply(tt_a, tt_b)
    out = _download(tt_out)

    assert_with_pcc(ref, out, PCC_ELTWISE)


# ---------------------------------------------------------------------------
# Activation functions: silu, gelu, relu
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    (1, 171, 6144),   # Qwen3 FFN gate
    (1, 17, 6144),    # action head FFN gelu
    (1, 16, 1536),    # ada silu
    (1, 1536),        # proj silu
    (1, 1, 1024),     # dec/state relu
    (1, 17, 1024),    # dec relu
])
def test_silu(shape, device):
    a = _rand(*shape)
    ref = torch.nn.functional.silu(a)

    tt_a = _upload(a, device)
    tt_out = ttnn.silu(tt_a)
    out = _download(tt_out)

    assert_with_pcc(ref, out, PCC_ELTWISE)


@pytest.mark.parametrize("shape", [
    (1, 576, 4304),   # SigLIP2 FFN fc1 gelu output
    (1, 144, 2048),   # MLP1 fc1 gelu output
    (1, 17, 6144),    # action head ff_gelu
])
def test_gelu(shape, device):
    a = _rand(*shape)
    ref = torch.nn.functional.gelu(a)

    tt_a = _upload(a, device)
    tt_out = ttnn.gelu(tt_a)
    out = _download(tt_out)

    assert_with_pcc(ref, out, PCC_ELTWISE)


@pytest.mark.parametrize("shape", [
    (1, 17, 1024),   # dec_relu / state_relu
    (1, 1, 1024),
])
def test_relu(shape, device):
    a = _rand(*shape)
    ref = torch.nn.functional.relu(a)

    tt_a = _upload(a, device)
    tt_out = ttnn.relu(tt_a)
    out = _download(tt_out)

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
    # Qwen3: (1,171,16,128) -> (1,16,171,128)
    ((1, 171, 16, 128), (0, 2, 1, 3)),
    # Qwen3 K transpose: (1,16,171,128) -> (1,16,128,171)
    ((1, 16, 171, 128), (0, 1, 3, 2)),
    # Action head: (1,17,32,48) -> (1,32,17,48)
    ((1, 17, 32, 48),   (0, 2, 1, 3)),
    # Action head K: (1,17,32,48) -> (1,32,48,17)
    ((1, 17, 32, 48),   (0, 2, 3, 1)),
    # Action head ctx: (1,32,17,48) -> (1,17,32,48)
    ((1, 32, 17, 48),   (0, 2, 1, 3)),
    # Action head cross-attn: (1,171,32,48) -> (1,32,171,48)
    ((1, 171, 32, 48),  (0, 2, 1, 3)),
    # Action head cross-attn K: (1,171,32,48) -> (1,32,48,171)
    ((1, 171, 32, 48),  (0, 2, 3, 1)),
])
def test_permute(shape, perm, device):
    a = _rand(*shape)
    ref = a.permute(perm).contiguous()

    tt_a = _upload(a, device)
    tt_out = ttnn.permute(tt_a, perm)
    out = _download(tt_out)

    assert_with_pcc(ref, out, PCC_ELTWISE)


# ---------------------------------------------------------------------------
# Reshape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_shape,out_shape", [
    # SigLIP2 patch reshape
    ((576, 1152),       (1, 576, 1152)),
    ((1, 576, 16, 72),  (1, 576, 1152)),
    ((1, 576, 1152),    (1, 576, 16, 72)),
    # MLP1 shuffle reshape (pixel-shuffle style)
    ((1, 576, 1152),    (1, 144, 4608)),
    # Qwen3 attention ctx reshape
    ((1, 171, 16, 128), (1, 171, 2048)),
    # Action head
    ((1, 17, 32, 48),   (1, 17, 1536)),
    ((1, 17, 1536),     (1, 17, 32, 48)),
    ((1, 1, 1536),      (1, 1536)),
    ((1, 1536),         (1, 1, 1536)),
    ((1, 16, 128),      (16, 128)),
    ((1, 16, 1536),     (16, 1536)),
    ((1, 16, 3072),     (16, 3072)),
    ((1, 1, 1024),      (1, 1024)),
    ((1, 1024),         (1, 1, 1024)),
    ((1, 1, 128),       (1, 128)),
])
def test_reshape(in_shape, out_shape, device):
    a = _rand(*in_shape)
    ref = a.reshape(out_shape)

    tt_a = _upload(a, device)
    tt_out = ttnn.reshape(tt_a, out_shape)
    out = _download(tt_out)

    assert_with_pcc(ref, out, PCC_ELTWISE)


# ---------------------------------------------------------------------------
# Concat
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shapes,dim", [
    # Action head: concat VL embeds + state embed along seq dim
    ([(1, 144, 2048), (1, 17, 2048)], 1),
    ([(1, 576, 1152), (1, 17, 1152)], 1),
    # concat along last dim
    ([(1, 17, 1024), (1, 17, 512)], -1),
])
def test_concat(shapes, dim, device):
    tensors = [_rand(*s) for s in shapes]
    ref = torch.cat(tensors, dim=dim)

    tt_tensors = [_upload(t, device) for t in tensors]
    tt_out = ttnn.concat(tt_tensors, dim=dim)
    out = _download(tt_out)

    assert_with_pcc(ref, out, PCC_ELTWISE)


# ---------------------------------------------------------------------------
# Repeat / broadcast
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape,repeats,dim", [
    # Action head: expand mask (1,1,1,171) -> (1,32,17,171)
    ((1, 1, 1, 171), [1, 32, 17, 1], None),
])
def test_repeat(shape, repeats, dim, device):
    a = _rand(*shape)
    ref = a.expand(repeats) if dim is None else a.repeat_interleave(repeats, dim=dim)

    tt_a = _upload(a, device)
    tt_out = ttnn.repeat(tt_a, ttnn.Shape(repeats))
    out = _download(tt_out)

    assert_with_pcc(ref, out, PCC_ELTWISE)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape,chunks,dim", [
    # Action head ada-split (1,3072) -> 2 × (1,1536)
    ((1, 3072),    2, -1),
    # Action head proj_out split (1,3072) -> 2 × (1,1536)
    ((1, 1536),    2, -1),
])
def test_split(shape, chunks, dim, device):
    a = _rand(*shape)
    ref_parts = torch.chunk(a, chunks, dim=dim)

    tt_a = _upload(a, device)
    tt_parts = ttnn.split(tt_a, a.shape[dim] // chunks, dim=dim)
    parts = [_download(p) for p in tt_parts]

    for ref, got in zip(ref_parts, parts):
        assert_with_pcc(ref, got, PCC_ELTWISE)
