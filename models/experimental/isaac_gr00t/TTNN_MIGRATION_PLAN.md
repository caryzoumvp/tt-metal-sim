# GR00T TTNN Migration Plan (Refined)

## Objective

Migrate GR00T inference to TTNN in 3 concrete steps:

1. Build input tokens from video dataset + text input on host CPU.
2. Implement vision-language backbone (VLM/Eagle path) with TTNN ops.
3. Implement diffusion transformer path with TTNN ops, using VLM output + robot state/action tokens as input.

PyTorch/Isaac-GR00T remains the oracle for every stage.

## Scope and Runtime Modes

- Keep only two runtime modes in demo:
  - `cpu_ref`: oracle compute and tensor dumps.
  - `ttnn_full`: TTNN compute + parity comparison vs oracle.
- Input source for migration validation:
  - Primary: `gr1` dataset sample (`traj_id`, `step`, `task_text` fixed per test case).
  - Secondary: `synthetic` for isolated debugging/perf smoke.

## System Boundary (for implementation)

- **Input builder (CPU)**:
  - dataset loading, video decoding, text prompt ingestion
  - processor/collator tokenization
  - assembled tensors for VLM + diffusion
- **VLM backbone (TTNN)**:
  - image/text token path
  - backbone hidden-state output (`backbone_features`, masks)
- **Diffusion transformer (TTNN)**:
  - state token encoder + action token encoder
  - concat with VLM context
  - denoise loop + action decoder

## Workstream 1: CPU Input Token Builder

### Deliverables

- Deterministic input assembly pipeline on host CPU:
  - video path, text, observation tensors
  - collated token tensors
  - System1/System2 input-shape dumps (already aligned with `minimal_gr00t_demo.py` style).
- Saved golden inputs for parity:
  - `input_ids`, `attention_mask`, `pixel_values`, `image_sizes`
  - `state`, `embodiment_id`
  - initial diffusion actions/noise seed

### Acceptance Criteria

- Same sample (`dataset`, `traj_id`, `step`, `task_text`) always produces identical assembled inputs under fixed seed.
- `cpu_ref` output block matches `minimal_gr00t_demo.py` format and key structure.

## Workstream 2: VLM Backbone TTNN (Eagle Path)

### Target

Replace PyTorch backbone forward with TTNN op runtime while keeping CPU input tokenization unchanged.

### TTNN Implementation Tasks

- Implement TTNN equivalents for backbone block stack:
  - QKV projections, attention, MLP, residual, normalization.
- Handle multimodal token/mask behavior:
  - image token mask
  - attention mask semantics identical to oracle.
- Produce oracle-compatible outputs:
  - `backbone_features`
  - `backbone_attention_mask`
  - `image_mask`

### Reference Code to Reuse

- `/workspaces/tensix/tt-metal/models/experimental/openvla/tt`
- `/workspaces/tensix/tt-metal/models/experimental/llama/tt`
- `/workspaces/tensix/tt-metal/models/experimental/gemma3_4b/tt`

### Acceptance Criteria

- Layer/block-level parity vs PyTorch backbone (max-abs + PCC thresholds).
- End output parity of `backbone_features` on fixed GR1 cases.

## Workstream 3: Diffusion Transformer TTNN

### Target

TTNN implementation of diffusion path that consumes:

- VLM outputs from Workstream 2.
- Robot state tokens.
- Action tokens.

### TTNN Implementation Tasks

- Keep existing TTNN action-head runtime contract and complete full-op path:
  - state encoder
  - action encoder
  - diffusion transformer blocks
  - output projection and action decoder
  - denoise loop integration
- Ensure exact input assembly:
  - `encoder_hidden_states` from VLM
  - `hidden_states = concat(state_tokens, action_tokens)`
  - timestep + masks

### Reference Code to Reuse

- `/workspaces/tensix/tt-metal/models/experimental/stable_diffusion_xl_base/tt`
- `/workspaces/tensix/tt-metal/models/experimental/pi0/reference`
- `/workspaces/tensix/tt-metal/models/experimental/openvla/tt`

### Acceptance Criteria

- Per-step denoise parity vs PyTorch diffusion path.
- Final action parity (per joint group) against `cpu_ref`.

## Integration Plan

### Phase A (Now)

- Freeze deterministic GR1 test vectors.
- Keep CPU input builder and CPU oracle stable.

### Phase B (System1 Backbone on TTNN)

Goal: move VLM backbone compute to TTNN in controlled slices, using replayed `prepared/system1` tensors and CPU oracle comparison at each slice.

#### B.0 Harness Baseline (done)

- `--backbone-only-compare`:
  - replay full CPU System1 from assembly dump and compare to saved `system1_output`.
- `--backbone-proj-compare --backbone-proj-layer <idx>`:
  - op-level compare for `self_attn.q_proj` (PyTorch vs TTNN matmul/fallback).

#### B.1 TTNN Linear Coverage (immediate next)

- Extend op-level compare from `q_proj` to:
  - `k_proj`, `v_proj`, `o_proj`
  - MLP projections (`gate/up/down` or model-equivalent names).
- Deliverable:
  - per-op diff report (max-abs, PCC) for each selected layer on fixed GR1 cases.

#### B.2 Single Decoder Layer TTNN

- Build one full TTNN decoder-layer replay path (start layer 12; then 13-15).
- Keep embeddings, mask prep, and non-migrated components in CPU.
- Deliverable:
  - `layer_output_tt` vs `layer_output_ref` parity dump.

#### B.3 Multi-layer Language Stack TTNN

- Chain migrated layers (12-15 first, then all language layers).
- Preserve mask semantics and rotary/position behavior exactly.
- Deliverable:
  - `backbone_features` parity on fixed GR1 matrix with visual path still CPU if needed.

#### B.4 Visual Path + Merge TTNN

- Migrate vision branch pieces used by Eagle path and multimodal merge/projection.
- Validate image-token positions and attention-mask semantics.
- Deliverable:
  - end-to-end System1 TTNN output parity:
    - `backbone_features`
    - `backbone_attention_mask`
    - `image_mask`

#### B Exit Gate

- `system1_mode=ttnn` path is stable on fixed GR1 cases.
- All System1 output tensors meet parity thresholds.
- System1 replay harness can run without invoking PyTorch backbone compute in critical path.

### Phase C (System2 Diffusion on TTNN)

Goal: migrate diffusion/action-head path to TTNN, consuming System1 TTNN outputs and preserving denoise-loop behavior.

#### C.0 Interface Freeze

- Freeze contract between System1 and System2:
  - `encoder_hidden_states`, `backbone_attention_mask`, `image_mask`
  - `state`, `embodiment_id`, action noise seed, timestep schedule.

#### C.1 Encoder Submodules on TTNN

- Move state encoder and action encoder to TTNN op path.
- Validate encoded tokens:
  - `state_features`
  - `action_features`
  - concatenated `sa_embs`.

#### C.2 Diffusion Block Migration

- Migrate DiT block stack incrementally:
  - attention path
  - feed-forward path
  - residual/norm/ada-modulation path.
- Validate block-by-block hidden states against CPU oracle.

#### C.3 Output Head + Denoise Loop

- Migrate output projection + action decoder.
- Run complete denoise loop on TTNN with fixed seed/timesteps.
- Validate per-step:
  - `pred_velocity`
  - `actions(t)` trajectory.

#### C.4 End-to-end `ttnn_full` Integration

- Use TTNN System1 output + TTNN System2 diffusion path in `ttnn_full`.
- Keep unified dump format for CPU/TTNN tensor maps.

#### C Exit Gate

- `ttnn_full` runs complete graph with dataset/text-driven inputs.
- Final decoded action parity passes per-key thresholds across validation matrix.
- No PyTorch compute remains in migrated critical path for System1/System2.

## Phase B/C Implementation Backlog (Ordered)

1. Add op-compare commands for System1 `k/v/o` + MLP projections.
2. Add single-layer TTNN replay command for one language decoder layer.
3. Add multi-layer TTNN replay command for tuned layers (12-15), then all layers.
4. Add System2 encoder parity command (`state_features`, `action_features`, `sa_embs`).
5. Add diffusion-block replay command with per-block hidden-state compare.
6. Add denoise-loop step compare command with trajectory dump.
7. Merge all into end-to-end `ttnn_full` execution path.

## Phase B/C Parity Gates

- Op-level gates:
  - max-abs diff
  - PCC
- Layer/block gates:
  - hidden-state parity at layer output and block boundaries.
- End-to-end gates:
  - per-step denoise parity
  - final per-key action parity (`left_arm`, `right_arm`, `left_hand`, `right_hand`, `waist`).

## Validation Matrix

- Cases:
  - GR1 dataset sample(s): at least 3 fixed (`traj_id`, `step`, `task_text`) combinations.
  - Two horizons: model default and reduced debug horizon.
- Metrics:
  - max-abs error
  - PCC
  - per-key action diff (`left_arm`, `right_arm`, `left_hand`, `right_hand`, `waist`)
- Artifacts:
  - tensor dumps for both paths (`cpu_ref`, `ttnn_full`)
  - concise diff summary in runtime logs.

## Exit Criteria

- `ttnn_full` runs end-to-end with dataset/text-driven inputs.
- VLM and diffusion subgraphs both on TTNN ops (no PyTorch compute in critical path).
- Action outputs pass agreed parity thresholds on fixed GR1 validation matrix.
- Regression command/scripts are stable and documented.

## Status Update (April 15, 2026)

### Completed (Phase A)

- Deterministic shared-input artifact flow implemented in:
  - `/workspaces/tensix/tt-metal/models/demos/isaac_gr00t/demo/gr00t.py`
- New runtime options:
  - `--save-shared-inputs-dir`
  - `--load-shared-inputs-dir`
  - `--dump-assembly-dir`
  - `--check-input-determinism`
- CPU assembly tensor dumps now include:
  - `observation`
  - `processor_output`
  - `collated_inputs`
  - `prepared/system1`
  - `prepared/system2`
  - `system1_output`
  - `shared_inputs`
- `cpu_ref` and `ttnn_full` now consume the same resolved shared-input path (either built or loaded), enabling repeatable parity checks.
- Added isolated System1 replay harness (`--backbone-only-compare`) to validate backbone parity from saved `prepared/system1` tensors before TTNN-op replacement.
- Added first System1 op-level migration harness:
  - `--backbone-proj-compare --backbone-proj-layer <idx>`
  - captures `self_attn.q_proj` input/output from replayed backbone graph
  - compares PyTorch vs TTNN matmul output (or CPU BF16 fallback when TT device is unavailable).

### In Progress (Phase B / Phase C)

- Phase B status:
  - completed in harness scope:
    - op-level compare extended to full attention + MLP linear set (layers 12-15).
    - linear-stack replay compare available via:
      - `--backbone-phaseb-compare --backbone-layer-start 12 --backbone-layer-end 15 --backbone-linear-set all`
    - module-level, decoder-layer-output, and System1-output parity checks integrated with tensor dumps.
  - pending:
    - run same Phase-B stack on real TT hardware (current environment has no detected TT chips).
- Phase C.0:
  - freeze and validate exact System1->System2 interface tensors for TTNN-only path.

### Implementation-First Note (April 15, 2026)

- Current priority is implementation completeness over simulator/hardware bring-up.
- Simulation-side end-to-end validation is intentionally deferred.
- During this phase:
  - complete code paths and parity-harness integration first
  - keep checks limited to static/syntax and host-side harness logic
  - postpone simulator/device debug to a dedicated follow-up stage.

### Phase C Implementation Status

- Implemented in code:
  - robust `ttnn_full` device-open fallback under `--no-strict-ttnn`
  - Phase-C gate summaries in runtime output (`C0 interface`, `C1 encoder`, `C2/C3 diffusion`, `C4 final`)
  - diffusion FFN dimension-aware path to support model variants in implementation mode.
  - standalone Phase-C parity commands:
    - `--system2-encoder-compare`
    - `--system2-block-compare --phasec-step <idx>`
    - `--system2-trajectory-compare`
- Pending (deferred by design):
  - simulator/hardware-backed end-to-end debug and performance tuning.
