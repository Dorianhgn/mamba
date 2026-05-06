#!/usr/bin/env python3
"""
Convert Mamba3SISOPortable to a CoreML stateful step model.

Produces:
  outputs/mamba3_siso_step.mlpackage  — autoregressive step, 4 StateType buffers (fp16)

State dtype: fp16 (required for ANE dispatch on iOS18)
Locked config: d_model=256, d_state=64, expand=2, headdim=64, rope_fraction=0.5

Run on dorian-mac (HDDLtorch conda env). Not executable on Linux (ct.convert is fine
on Linux, but predict() with CPU_ONLY requires macOS coremltools runtime).
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR))
from mamba3_siso_portable import Mamba3SISOPortable

CONFIG = dict(
    d_model=256, d_state=64, expand=2, headdim=64,
    rope_fraction=0.5, A_floor=1e-4, rms_eps=1e-5,
)

OUTPUTS_DIR = _TESTS_DIR.parent / "outputs"
GOLDEN_PATH = _TESTS_DIR / "golden.pt"

# State shapes for batch=1, locked config:
#   nheads = (expand * d_model) // headdim = (2*256)//64 = 8
#   num_rope_angles = int(d_state * rope_fraction) // 2 = 16
B               = 1
NHEADS          = 8
NUM_ROPE_ANGLES = 16
HEADDIM         = 64
D_STATE         = 64
D_MODEL         = 256


class StepWrapper(nn.Module):
    """
    CoreML-traceable single-step wrapper with stateful fp16 buffers.

    All 4 recurrent states are fp16 registered buffers so CoreML treats them
    as StateType tensors (iOS18). In-place copy_ on each buffer after step()
    is the pattern CoreML uses to detect and persist state updates.
    """
    def __init__(self, model: Mamba3SISOPortable):
        super().__init__()
        self.m = model
        self.register_buffer("angle_state", torch.zeros(B, NHEADS, NUM_ROPE_ANGLES,     dtype=torch.float16))
        self.register_buffer("ssm_state",   torch.zeros(B, NHEADS, HEADDIM, D_STATE,    dtype=torch.float16))
        self.register_buffer("k_state",     torch.zeros(B, 1, NHEADS, D_STATE,          dtype=torch.float16))
        self.register_buffer("v_state",     torch.zeros(B, NHEADS, HEADDIM,             dtype=torch.float16))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        out, new_angle, new_ssm, new_k, new_v = self.m.step(
            u, self.angle_state, self.ssm_state, self.k_state, self.v_state
        )
        # Slice assignment forces aten::slice into the TorchScript IR so that
        # coremltools generate_tensor_assignment_ops finds the
        # GetAttr → slice → copy_(cast) pattern required for coreml_update_state.
        #
        # coremltools also needs an aten::to cast node immediately before copy_.
        # new_angle (fp32 theta) and new_k (fp32 K_rot) always produce real casts.
        # new_ssm keeps its cast node alive after inlining.
        # new_v = x_raw.reshape() is already fp16 — the .to(fp16) no-op is
        # optimised away, so coreml_update_state is NOT generated for v_state.
        # Fix: force a genuine fp16→fp32→fp16 round-trip for v_state.
        self.angle_state[:] = new_angle.to(torch.float16)
        self.ssm_state[:] = new_ssm.to(torch.float16)
        self.k_state[:] = new_k.to(torch.float16)
        # v_state: new_v = x_raw.reshape() has NO data-flow path from self.v_state,
        # so TorchScript DCE eliminates the write as a dead side effect.
        # Fix: add self.v_state * 0 to create a graph edge from the read to the write
        # value, preserving the side-effectful copy_ in the traced IR.
        self.v_state[:] = (new_v + self.v_state * 0).float().half()
        return out.to(torch.float16)


def load_model() -> Mamba3SISOPortable:
    """Load model from golden.pt state_dict (fp32 weights → half). Falls back to random weights."""
    if GOLDEN_PATH.exists():
        data = torch.load(GOLDEN_PATH, map_location="cpu", weights_only=False)
        state_dict = data.get("state_dict")
        if state_dict is not None:
            model = Mamba3SISOPortable(**CONFIG, dtype=torch.float32)
            model.load_state_dict(state_dict)
            model.half().eval()
            print(f"[load] Loaded weights from {GOLDEN_PATH} (source: {data.get('source', '?')})")
            return model
    torch.manual_seed(42)
    print("[load] WARNING: no golden.pt state_dict found; using random fp16 weights")
    return Mamba3SISOPortable(**CONFIG, dtype=torch.float16).eval()


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "mamba3_siso_step.mlpackage"

    print("=== Building StepWrapper ===")
    model = load_model()
    wrapper = StepWrapper(model).eval()

    print("=== Tracing (sample input: zeros fp16) ===")
    sample_u = torch.zeros(B, D_MODEL, dtype=torch.float16)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, sample_u)
    print("[trace] OK")

    # Print TorchScript IR summary for debugging
    graph_str = str(traced.graph)
    # Show all copy_ and GetAttr lines for states
    print("[ir] copy_ and state-GetAttr lines:")
    for l in graph_str.split('\n'):
        if 'copy_' in l or ('GetAttr' in l and 'state' in l):
            print("  ", l.strip()[:120])
    # Show all lines for each state variable
    for state in ['angle_state', 'ssm_state', 'k_state', 'v_state']:
        lines = [l for l in graph_str.split('\n') if state in l]
        print(f"[ir] {state}: {len(lines)} lines")

    print("=== Converting to CoreML (iOS18, fp16, stateful) ===")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="u", shape=(B, D_MODEL), dtype=np.float16)],
        outputs=[ct.TensorType(name="out", dtype=np.float16)],
        states=[
            ct.StateType(
                wrapped_type=ct.TensorType(shape=(B, NHEADS, NUM_ROPE_ANGLES),    dtype=np.float16),
                name="angle_state",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=(B, NHEADS, HEADDIM, D_STATE),   dtype=np.float16),
                name="ssm_state",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=(B, 1, NHEADS, D_STATE),          dtype=np.float16),
                name="k_state",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=(B, NHEADS, HEADDIM),             dtype=np.float16),
                name="v_state",
            ),
        ],
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
    )
    print("[convert] OK")

    # Inspect MIL before saving — check coreml_update_state generated for all 4 states
    prog = mlmodel._mil_program
    if prog is not None:
        main_fn = prog.functions["main"]
        update_ops = [op for op in main_fn.operations if op.op_type == "coreml_update_state"]
        print(f"[mil] coreml_update_state ops: {len(update_ops)}")
        for op in update_ops:
            print(f"  state={op.inputs.get('state')}  value={op.inputs.get('value')}")
    else:
        print("[mil] _mil_program not available")

    print(f"=== Saving to {out_path} ===")
    mlmodel.save(str(out_path))
    print(f"[save] OK — {out_path}")

    # Quick sanity: load with CPU_ONLY and run 1 step
    print("=== Sanity check (CPU_ONLY, 1 step) ===")
    loaded = ct.models.MLModel(str(out_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    state = loaded.make_state()
    result = loaded.predict({"u": np.zeros((B, D_MODEL), dtype=np.float16)}, state=state)
    out_shape = result["out"].shape
    print(f"[sanity] out shape: {out_shape} — {'OK' if out_shape[-1] == D_MODEL else 'UNEXPECTED'}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
