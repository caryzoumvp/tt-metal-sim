# Isaac-GR00T TTNN Migration Demo

## How to Run

This demo is TTNN-native and does not execute Isaac-GR00T runtime code.
It directly loads local checkpoint weights from `GR00T-N1.6-3B`.

Create environment:

```bash
cd /workspaces/tensix/tt-metal/models/demos/isaac_gr00t
conda env create -f environment.yaml
conda activate tt-isaac-gr00t
```

From the `tt-metal` repo root:

```bash
python models/demos/isaac_gr00t/demo/gr00t.py \
  --tt-device-id 0
```

Choose embodiment id (optional):

```bash
python models/demos/isaac_gr00t/demo/gr00t.py \
  --model-dir /workspaces/tensix/tt-metal/models/demos/isaac_gr00t/GR00T-N1.6-3B \
  --embodiment-id 0 \
  --action-horizon 8 \
  --num-inference-timesteps 4 \
  --tt-device-id 0
```

List candidate weight keys:

```bash
python models/demos/isaac_gr00t/demo/gr00t.py \
  --model-dir /workspaces/tensix/tt-metal/models/demos/isaac_gr00t/GR00T-N1.6-3B \
  --list-keys
```

## Notes

- The script maps real GR00T action-head weights:
  - `state_encoder`, `action_encoder`, timestep MLP
  - attention block-0 `to_q/to_k/to_v/to_out`
  - `proj_out_2` and `action_decoder`
- The script prints per-op tensor shapes, including attention and diffusion-loop steps.
- It reports max absolute difference between TTNN decoder output and CPU decoder output.
- Supported checkpoint formats: `model.safetensors(.index.json)` and `pytorch_model.bin`.
