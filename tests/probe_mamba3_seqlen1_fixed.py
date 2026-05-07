"""Verify mamba3.py fix: state must differ between token-with-prior-state and token-with-fresh-state."""
import sys, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from mamba_ssm.modules.mamba3 import Mamba3
try:
    from mamba_ssm.utils.generation import InferenceParams
except ImportError:
    from dataclasses import dataclass, field
    @dataclass
    class InferenceParams:
        max_seqlen: int; max_batch_size: int; seqlen_offset: int = 0
        batch_size_offset: int = 0; key_value_memory_dict: dict = field(default_factory=dict)
        lengths_per_sample = None

CFG = dict(d_model=512, d_state=64, expand=1, headdim=64,
           ngroups=1, rope_fraction=0.5, is_mimo=False,
           is_outproj_norm=False, layer_idx=0, device='cuda', dtype=torch.float32)
torch.manual_seed(42)
model = Mamba3(**CFG).cuda().float().eval()
tokens = torch.randn(3, 1, 1, 512, device='cuda')

# Path A: token0 then token1 (state carried)
inf_a = InferenceParams(max_seqlen=4, max_batch_size=1)
with torch.no_grad():
    model(tokens[0], inference_params=inf_a)
    out1_carried = model(tokens[1], inference_params=inf_a).squeeze()

# Path B: fresh state, token1 only
inf_b = InferenceParams(max_seqlen=4, max_batch_size=1)
with torch.no_grad():
    out1_fresh = model(tokens[1], inference_params=inf_b).squeeze()

diff = (out1_carried - out1_fresh).abs().max().item()
assert diff > 0, f"FAIL: state not carried (diff={diff})"
print(f"PASS: diff={diff:.4e} > 0  (state IS carried across seqlen=1 calls)")
