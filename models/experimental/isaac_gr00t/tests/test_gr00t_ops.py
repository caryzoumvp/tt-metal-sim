# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Op-level unit tests for GR00T N1.6 TTNN kernels.

This file is table-driven from:
    /workspaces/wormhole_sim/m5out_gr00t_sim_0610/profile_mark_host_labels.csv

Each row in PROFILE_CASES is one profiled host label. The test dispatches the
matching TTNN primitive with the same input shapes and compares against a torch
reference. In --sim mode, sim_timer records the original op_name and shapes so
the generated labels line up with the source profile.

Run:
    cd /workspaces/tensix/tt-metal
    pytest models/experimental/isaac_gr00t/tests/test_gr00t_ops.py -v
    pytest models/experimental/isaac_gr00t/tests/test_gr00t_ops.py -v --sim
"""

import itertools
import math

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


PCC_MATMUL = 0.98
PCC_ELTWISE = 0.99
RMS_NORM_EPS = 1e-6
SIGLIP_HEAD_DIM = 72
QWEN_HEAD_DIM = 128


PROFILE_CASES = [
    (1, "sg.patch.linear", [(576, 588), (588, 1152)], (576, 1152)),
    (2, "sg.patch.reshape", [(576, 1152)], (1, 576, 1152)),
    (3, "sg.pos.add", [(1, 576, 1152), (1, 576, 1152)], (1, 576, 1152)),
    (4, "sg.00.ln1.layer_norm", [(1, 576, 1152)], (1, 576, 1152)),
    (5, "sg.00.attn.q.linear", [(1, 576, 1152), (1152, 1152)], (1, 576, 1152)),
    (6, "sg.00.attn.k.linear", [(1, 576, 1152), (1152, 1152)], (1, 576, 1152)),
    (7, "sg.00.attn.v.linear", [(1, 576, 1152), (1152, 1152)], (1, 576, 1152)),
    (8, "sg.00.attn.q.reshape", [(1, 576, 1152)], (1, 576, 16, 72)),
    (9, "sg.00.attn.q.perm", [(1, 576, 16, 72)], (1, 16, 576, 72)),
    (10, "sg.00.attn.k.reshape", [(1, 576, 1152)], (1, 576, 16, 72)),
    (11, "sg.00.attn.k.perm", [(1, 576, 16, 72)], (1, 16, 576, 72)),
    (12, "sg.00.attn.v.reshape", [(1, 576, 1152)], (1, 576, 16, 72)),
    (13, "sg.00.attn.v.perm", [(1, 576, 16, 72)], (1, 16, 576, 72)),
    (14, "sg.00.attn.k.t", [(1, 16, 576, 72)], (1, 16, 72, 576)),
    (15, "sg.00.attn.qk.matmul", [(1, 16, 576, 72), (1, 16, 72, 576)], (1, 16, 576, 576)),
    (16, "sg.00.attn.qk.scale", [(1, 16, 576, 576)], (1, 16, 576, 576)),
    (17, "sg.00.attn.softmax", [(1, 16, 576, 576)], (1, 16, 576, 576)),
    (18, "sg.00.attn.pv.matmul", [(1, 16, 576, 576), (1, 16, 576, 72)], (1, 16, 576, 72)),
    (19, "sg.00.attn.ctx.perm", [(1, 16, 576, 72)], (1, 576, 16, 72)),
    (20, "sg.00.attn.ctx.reshape", [(1, 576, 16, 72)], (1, 576, 1152)),
    (21, "sg.00.attn.o.linear", [(1, 576, 1152), (1152, 1152)], (1, 576, 1152)),
    (22, "sg.00.attn.res", [(1, 576, 1152), (1, 576, 1152)], (1, 576, 1152)),
    (23, "sg.00.ln2.layer_norm", [(1, 576, 1152)], (1, 576, 1152)),
    (24, "sg.00.ffn.fc1.linear_gelu", [(1, 576, 1152), (1152, 4304)], (1, 576, 4304)),
    (25, "sg.00.ffn.fc2.linear", [(1, 576, 4304), (4304, 1152)], (1, 576, 1152)),
    (26, "sg.00.ffn.res", [(1, 576, 1152), (1, 576, 1152)], (1, 576, 1152)),
    (27, "sg.post_ln.layer_norm", [(1, 576, 1152)], (1, 576, 1152)),
    (28, "mlp1.ln.layer_norm", [(1, 144, 4608)], (1, 144, 4608)),
    (29, "mlp1.fc1.linear_gelu", [(1, 144, 4608), (4608, 2048)], (1, 144, 2048)),
    (30, "mlp1.fc2.linear", [(1, 144, 2048), (2048, 2048)], (1, 144, 2048)),
    (31, "qw.00.ln1.rms_norm", [(1, 171, 2048)], (1, 171, 2048)),
    (32, "qw.00.attn.q.linear", [(1, 171, 2048), (2048, 2048)], (1, 171, 2048)),
    (33, "qw.00.attn.k.linear", [(1, 171, 2048), (2048, 1024)], (1, 171, 1024)),
    (34, "qw.00.attn.v.linear", [(1, 171, 2048), (2048, 1024)], (1, 171, 1024)),
    (35, "qw.00.attn.sdpa", [(1, 16, 171, 128), (1, 16, 171, 128), (1, 16, 171, 128)], (1, 16, 171, 128)),
    (36, "qw.00.attn.ctx.concat_heads", [(1, 16, 171, 128)], (1, 1, 171, 2048)),
    (37, "qw.00.attn.ctx.reshape", [(1, 1, 171, 2048)], (1, 171, 2048)),
    (38, "qw.00.attn.o.linear", [(1, 171, 2048), (2048, 2048)], (1, 171, 2048)),
    (39, "qw.00.attn.res", [(1, 171, 2048), (1, 171, 2048)], (1, 171, 2048)),
    (40, "qw.00.ln2.rms_norm", [(1, 171, 2048)], (1, 171, 2048)),
    (41, "qw.00.ffn.gate.linear", [(1, 171, 2048), (2048, 6144)], (1, 171, 6144)),
    (42, "qw.00.ffn.gate.silu", [(1, 171, 6144)], (1, 171, 6144)),
    (43, "qw.00.ffn.up.linear", [(1, 171, 2048), (2048, 6144)], (1, 171, 6144)),
    (44, "qw.00.ffn.gate_up.mul", [(1, 171, 6144), (1, 171, 6144)], (1, 171, 6144)),
    (45, "qw.00.ffn.down.linear", [(1, 171, 6144), (6144, 2048)], (1, 171, 2048)),
    (46, "qw.00.ffn.res", [(1, 171, 2048), (1, 171, 2048)], (1, 171, 2048)),
    (47, "qw.final_norm.rms_norm", [(1, 171, 2048)], (1, 171, 2048)),
]


def _rand(shape, dtype=torch.bfloat16):
    return torch.randn(shape, dtype=dtype)


def _upload(t, device):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)


def _download(t):
    return ttnn.to_torch(t)


def _case_id(case):
    seq_num, op_name, in_shapes, _ = case
    return f"{seq_num:02d}-{op_name}-{in_shapes}"


def _infer_permutation(in_shape, out_shape):
    for perm in itertools.permutations(range(len(in_shape))):
        if tuple(in_shape[i] for i in perm) == tuple(out_shape):
            return perm
    raise ValueError(f"cannot infer permutation from {in_shape} to {out_shape}")


def _rms_norm_ref(x, weight, eps=RMS_NORM_EPS):
    x_float = x.float()
    out = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (out * weight.float()).to(torch.bfloat16)


def _run_linear(op_name, tensors, device):
    a, w = tensors
    activation = "gelu" if op_name.endswith(".linear_gelu") else None
    ref = torch.matmul(a, w)
    if activation == "gelu":
        ref = torch.nn.functional.gelu(ref)

    tt_a = _upload(a, device)
    tt_w = _upload(w, device)
    if activation is None:
        return ref, lambda: _download(ttnn.linear(tt_a, tt_w))
    return ref, lambda: _download(ttnn.linear(tt_a, tt_w, activation=activation))


def _run_layer_norm(tensors, device):
    (a,) = tensors
    weight = torch.ones((a.shape[-1],), dtype=torch.bfloat16)
    bias = torch.zeros((a.shape[-1],), dtype=torch.bfloat16)
    ref = torch.nn.functional.layer_norm(a, (a.shape[-1],), weight=weight, bias=bias)

    tt_a = _upload(a, device)
    tt_weight = _upload(weight, device)
    tt_bias = _upload(bias, device)
    return ref, lambda: _download(ttnn.layer_norm(tt_a, weight=tt_weight, bias=tt_bias))


def _run_rms_norm(tensors, device):
    (a,) = tensors
    weight = torch.ones((1, a.shape[-1]), dtype=torch.bfloat16)
    ref = _rms_norm_ref(a, weight)

    tt_a = _upload(a, device)
    tt_weight = _upload(weight, device)
    return ref, lambda: _download(ttnn.rms_norm(tt_a, weight=tt_weight, epsilon=RMS_NORM_EPS))


def _run_sdpa(tensors, device):
    q, k, v = tensors
    scale = 1.0 / math.sqrt(QWEN_HEAD_DIM)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=True, scale=scale
    ).to(torch.bfloat16)

    tt_q = _upload(q, device)
    tt_k = _upload(k, device)
    tt_v = _upload(v, device)
    return ref, lambda: _download(
        ttnn.transformer.scaled_dot_product_attention(tt_q, tt_k, tt_v, is_causal=True, scale=scale)
    )


def _build_reference_and_runner(op_name, in_shapes, out_shape, device):
    tensors = [_rand(shape) for shape in in_shapes]

    if op_name.endswith(".linear") or op_name.endswith(".linear_gelu"):
        return _run_linear(op_name, tensors, device)

    if op_name.endswith(".reshape"):
        (a,) = tensors
        ref = a.reshape(out_shape)
        tt_a = _upload(a, device)
        return ref, lambda: _download(ttnn.reshape(tt_a, out_shape))

    if op_name.endswith(".perm") or op_name.endswith(".t"):
        (a,) = tensors
        perm = _infer_permutation(in_shapes[0], out_shape)
        ref = a.permute(perm).contiguous()
        tt_a = _upload(a, device)
        return ref, lambda: _download(ttnn.permute(tt_a, perm))

    if op_name.endswith(".add") or op_name.endswith(".res"):
        a, b = tensors
        ref = a + b
        tt_a = _upload(a, device)
        tt_b = _upload(b, device)
        return ref, lambda: _download(ttnn.add(tt_a, tt_b))

    if op_name.endswith(".matmul"):
        a, b = tensors
        ref = torch.matmul(a, b)
        tt_a = _upload(a, device)
        tt_b = _upload(b, device)
        return ref, lambda: _download(ttnn.matmul(tt_a, tt_b))

    if op_name.endswith(".scale"):
        (a,) = tensors
        scale = 1.0 / math.sqrt(SIGLIP_HEAD_DIM)
        ref = a * scale
        tt_a = _upload(a, device)
        return ref, lambda: _download(ttnn.multiply(tt_a, scale))

    if op_name.endswith(".mul"):
        a, b = tensors
        ref = a * b
        tt_a = _upload(a, device)
        tt_b = _upload(b, device)
        return ref, lambda: _download(ttnn.multiply(tt_a, tt_b))

    if op_name.endswith(".softmax"):
        (a,) = tensors
        ref = torch.softmax(a.float(), dim=-1).to(torch.bfloat16)
        tt_a = _upload(a, device)
        return ref, lambda: _download(ttnn.softmax(tt_a, dim=-1))

    if op_name.endswith(".layer_norm"):
        return _run_layer_norm(tensors, device)

    if op_name.endswith(".rms_norm"):
        return _run_rms_norm(tensors, device)

    if op_name.endswith(".silu"):
        (a,) = tensors
        ref = torch.nn.functional.silu(a)
        tt_a = _upload(a, device)
        return ref, lambda: _download(ttnn.silu(tt_a))

    if op_name.endswith(".sdpa"):
        return _run_sdpa(tensors, device)

    if op_name.endswith(".concat_heads"):
        (a,) = tensors
        batch, heads, seq_len, head_dim = a.shape
        ref = a.permute(0, 2, 1, 3).contiguous().reshape(batch, 1, seq_len, heads * head_dim)
        tt_a = _upload(a, device)
        return ref, lambda: _download(ttnn.experimental.nlp_concat_heads(tt_a))

    raise ValueError(f"unsupported profiled op: {op_name}")


@pytest.mark.parametrize("case", PROFILE_CASES, ids=_case_id)
def test_profiled_gr00t_op(case, device, sim_timer):
    _, op_name, in_shapes, out_shape = case
    ref, run_ttnn = _build_reference_and_runner(op_name, in_shapes, out_shape, device)

    out = sim_timer(op_name, run_ttnn, in_shapes=str(in_shapes), out_shape=str(out_shape))

    assert tuple(out.shape) == tuple(out_shape)
    matmul_like = any(op_name.endswith(suffix) for suffix in (".linear", ".linear_gelu", ".matmul", ".sdpa"))
    assert_with_pcc(ref, out, PCC_MATMUL if matmul_like else PCC_ELTWISE)
