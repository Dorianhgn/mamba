"""
Mamba3SISOPortable — pure-PyTorch SISO step() for Core ML portability.

Implements the 8 stages from docs/2.1_spec.md exactly:
  1. in_proj + split
  2. A, Δt, λ (data-dependent scalars)
  3. B/C reshape → RMSNorm → broadcast → bias
  4. Δθ accumulation + pairwise RoPE on K, Q
  5. α, β, γ coefficients
  6. SSM recurrence h = α·h + β·(V_{t-1}⊗K_{t-1}) + γ·(V_t⊗K_t)
  7. y = h·Q + D·V, gated by SiLU(z)
  8. out_proj

Constraints:
  - No triton / tilelang / cute / quack imports.
  - No hardcoded .float() or .to(torch.float32) in the main computation path.
  - States passed explicitly; nothing in inference_params.
  - Works for fp32 and fp16 model dtypes.

DEVIATION notes are inline with # DEVIATION: tags.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Plain RMSNorm matching the behavior of RMSNormGated (without gate).
    Computes variance in float32 internally for stability, returns input dtype.
    """
    def __init__(self, dim: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        rms_inv = (x_f.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        return (x_f * rms_inv * self.weight.float()).to(orig_dtype)


def rope_pairwise(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    """
    Pairwise RoPE: rotate pairs (x_{2i}, x_{2i+1}) for i in [0, rotary_dim//2).
    Matches ROTATE_PAIRWISE=True in mamba3_mimo_rotary_step.py.

    Args:
        x:   (..., d_state)
        cos: (..., rotary_dim//2)  — broadcast-compatible with x leading dims
        sin: (..., rotary_dim//2)
        rotary_dim: number of dimensions to rotate (must be even, ≤ d_state)
    Returns:
        rotated tensor in the dtype of the computation (fp32 when cos/sin are fp32).
        For the locked config (rope_fraction=0.5, d_state=64), rotary_dim=32 < d_state,
        so the non-rotated pass-through half is cast to the output dtype via cat.
    """
    x_rot  = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]   # may be empty if rotary_dim == x.shape[-1]

    # Reshape last dim to pairs: (..., rotary_dim//2, 2)
    x_rot = x_rot.reshape(x_rot.shape[:-1] + (rotary_dim // 2, 2))
    x0, x1 = x_rot[..., 0], x_rot[..., 1]   # (..., rotary_dim//2)

    xo0 = x0 * cos - x1 * sin   # promoted to fp32 when cos is fp32
    xo1 = x0 * sin + x1 * cos

    # Interleave pairs back: (..., rotary_dim//2, 2) → (..., rotary_dim)
    x_out = torch.stack([xo0, xo1], dim=-1).flatten(-2)

    if rotary_dim < x.shape[-1]:
        # x_pass may be fp16 while x_out is fp32 (promoted by fp32 cos/sin).
        # Cast x_pass to x_out's dtype so torch.cat doesn't raise.
        x_out = torch.cat([x_out, x_pass.to(x_out.dtype)], dim=-1)

    return x_out


# ---------------------------------------------------------------------------
# Main portable module
# ---------------------------------------------------------------------------

class Mamba3SISOPortable(nn.Module):
    """
    Pure-PyTorch SISO Mamba3 implementing step() and forward().

    Public API matches the original Mamba3 (SISO, is_outproj_norm=False).

    Args (locked config from docs/mamba3_iphone_plan.md §1):
        d_model      = 256
        d_state      = 64
        expand       = 2     (d_inner = 512)
        headdim      = 64    (nheads = 8)
        rope_fraction= 0.5   (rotary_dim = 32, num_rope_angles = 16)
        A_floor      = 1e-4
        rms_eps      = 1e-5
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        rope_fraction: float = 0.5,
        A_floor: float = 1e-4,
        rms_eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model      = d_model
        self.d_state      = d_state
        self.expand       = expand
        self.headdim      = headdim
        self.A_floor      = A_floor

        self.d_inner = int(expand * d_model)
        assert self.d_inner % headdim == 0
        self.nheads = self.d_inner // headdim

        # SISO-only (mimo_rank=1, num_bc_heads=1 per locked config)
        self.mimo_rank    = 1
        self.num_bc_heads = 1

        # RoPE dims  [§1 / §3]
        assert rope_fraction in (0.5, 1.0)
        split_tensor_size = int(d_state * rope_fraction)
        if split_tensor_size % 2 != 0:
            split_tensor_size -= 1
        self.num_rope_angles = split_tensor_size // 2
        self.rotary_dim      = split_tensor_size  # = 2 * num_rope_angles

        # in_proj split sizes [§1]
        self._split = [
            self.d_inner,          # z
            self.d_inner,          # x  (= V)
            d_state,               # B  (= K raw)
            d_state,               # C  (= Q raw)
            self.nheads,           # dd_dt
            self.nheads,           # dd_A
            self.nheads,           # trap
            self.num_rope_angles,  # angles
        ]
        d_in_proj = sum(self._split)

        self.in_proj  = nn.Linear(d_model, d_in_proj, bias=False, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **factory_kwargs)

        # dt_bias [§2.1]
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads, **factory_kwargs))
        self.dt_bias._no_weight_decay = True

        # B / C biases: (nheads, mimo_rank=1, d_state) [§3]
        self.B_bias = nn.Parameter(
            torch.ones(self.nheads, 1, d_state, **factory_kwargs)
        )
        self.C_bias = nn.Parameter(
            torch.ones(self.nheads, 1, d_state, **factory_kwargs)
        )

        # RMSNorm for B and C [§2.2]
        self.B_norm = RMSNorm(d_state, eps=rms_eps, device=device, dtype=dtype)
        self.C_norm = RMSNorm(d_state, eps=rms_eps, device=device, dtype=dtype)

        # D skip [§5]
        self.D = nn.Parameter(torch.ones(self.nheads, **factory_kwargs))
        self.D._no_weight_decay = True

    # -----------------------------------------------------------------------
    # Weight loading
    # -----------------------------------------------------------------------

    def load_from_original(self, mamba3_module: nn.Module) -> None:
        """
        Copy weights from an original Mamba3 instance (SISO, is_outproj_norm=False).
        Handles dtype / device differences transparently via copy_().
        """
        with torch.no_grad():
            self.in_proj.weight.copy_(mamba3_module.in_proj.weight)
            self.out_proj.weight.copy_(mamba3_module.out_proj.weight)
            self.dt_bias.copy_(mamba3_module.dt_bias)
            self.D.copy_(mamba3_module.D)
            self.B_bias.copy_(mamba3_module.B_bias)
            self.C_bias.copy_(mamba3_module.C_bias)
            self.B_norm.weight.copy_(mamba3_module.B_norm.weight)
            self.C_norm.weight.copy_(mamba3_module.C_norm.weight)

    # -----------------------------------------------------------------------
    # State allocation
    # -----------------------------------------------------------------------

    def allocate_states(
        self,
        batch_size: int,
        device=None,
        dtype=None,
    ):
        """
        Allocate zero-initialised inference states.
        Matches original allocate_inference_cache():
          - angle_state, ssm_state: float32 (always)
          - k_state, v_state: model dtype
        """
        device = self.in_proj.weight.device if device is None else device
        dtype  = self.in_proj.weight.dtype  if dtype  is None else dtype

        angle_state = torch.zeros(
            batch_size, self.nheads, self.num_rope_angles,
            device=device, dtype=torch.float32,
        )
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state,
            device=device, dtype=torch.float32,
        )
        k_state = torch.zeros(
            batch_size, 1, self.nheads, self.d_state,
            device=device, dtype=dtype,
        )
        v_state = torch.zeros(
            batch_size, self.nheads, self.headdim,
            device=device, dtype=dtype,
        )
        return angle_state, ssm_state, k_state, v_state

    # -----------------------------------------------------------------------
    # Core computation
    # -----------------------------------------------------------------------

    def _compute_step(
        self,
        u: torch.Tensor,
        angle_state: torch.Tensor,
        ssm_state: torch.Tensor,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        capture: dict = None,
    ):
        """
        Single-step computation with optional tensor capture for parity testing.
        All stages reference the spec (docs/2.1_spec.md §N).

        Args:
            u:           (B, d_model)
            angle_state: (B, nheads, num_rope_angles)  float32
            ssm_state:   (B, nheads, headdim, d_state)  float32
            k_state:     (B, 1, nheads, d_state)
            v_state:     (B, nheads, headdim)
            capture:     optional dict; filled in-place with intermediate tensors
        Returns:
            out, new_angle_state, new_ssm_state, new_k_state, new_v_state
        """
        def cap(name: str, val: torch.Tensor) -> torch.Tensor:
            if capture is not None:
                capture[name] = val.detach().clone()
            return val

        B = u.shape[0]

        # ── Stage 1: in_proj + split  [§1] ──────────────────────────────────
        zxBCdt = self.in_proj(u)
        cap('in_proj_out', zxBCdt)

        z_raw, x_raw, B_raw, C_raw, dd_dt, dd_A, trap_raw, angles_raw = \
            torch.split(zxBCdt, self._split, dim=-1)

        cap('z_split',      z_raw)
        cap('x_split',      x_raw)
        cap('B_split',      B_raw)
        cap('C_split',      C_raw)
        cap('dd_dt_split',  dd_dt)
        cap('dd_A_split',   dd_A)
        cap('trap_split',   trap_raw)
        cap('angles_split', angles_raw)

        # ── Stage 2: data-dependent scalar activations  [§2.1] ──────────────
        # A = -max(softplus(dd_A), A_floor), using clamp(x, max=-A_floor) on -softplus
        # Note: no hardcoded .float() — softplus operates in input dtype.
        A   = -torch.clamp(F.softplus(dd_A), min=self.A_floor)   # (B, nheads)
        DT  = F.softplus(dd_dt + self.dt_bias)                    # (B, nheads)
        lam = torch.sigmoid(trap_raw)                              # (B, nheads)

        cap('A_postclamp', A)
        cap('DT',          DT)
        cap('lambda',      lam)

        # ── Stage 3: reshape → RMSNorm → broadcast → bias  [§2.2] ──────────
        # SISO: mimo_rank=1, num_bc_heads=1 → reshape to (B, 1, 1, d_state)
        Kn = B_raw.reshape(B, 1, 1, self.d_state)
        Qn = C_raw.reshape(B, 1, 1, self.d_state)

        Kn = self.B_norm(Kn)   # RMSNorm over last dim (d_state)
        Qn = self.C_norm(Qn)
        cap('B_norm_out', Kn)
        cap('C_norm_out', Qn)

        # Broadcast to all nheads: (B, 1, nheads, d_state)
        Kn = Kn.expand(-1, -1, self.nheads, -1)
        Qn = Qn.expand(-1, -1, self.nheads, -1)
        cap('B_bcast', Kn)
        cap('C_bcast', Qn)

        # Add per-head biases (rearranged to (1, nheads, d_state) for broadcast)
        # B_bias shape: (nheads, 1, d_state) → rearrange "h r n -> r h n" → (1, nheads, d_state)
        K_bias = self.B_bias.transpose(0, 1)   # (1, nheads, d_state)
        Q_bias = self.C_bias.transpose(0, 1)   # (1, nheads, d_state)

        K_pre_rope = Kn + K_bias[None]         # (B, 1, nheads, d_state)
        Q_pre_rope = Qn + Q_bias[None]
        cap('K_pre_rope', K_pre_rope)
        cap('Q_pre_rope', Q_pre_rope)

        # ── Stage 4: angle accumulation + pairwise RoPE  [§3] ───────────────
        # angles_raw: (B, num_rope_angles) → (B, nheads, num_rope_angles)
        angles = angles_raw.unsqueeze(1).expand(-1, self.nheads, -1)

        # Δθ = tanh(angles) * π * Δt   [§3 i]
        delta_theta = torch.tanh(angles) * torch.pi * DT.unsqueeze(-1)
        cap('delta_theta', delta_theta)

        # θ = angle_state + Δθ   [§3 ii]
        # DEVIATION: spec states mod 2π but actual kernel implementation does NOT apply
        # modulo — cos/sin are periodic so it's numerically equivalent for bounded inputs.
        theta = angle_state + delta_theta          # (B, nheads, num_rope_angles)
        # angle_state is fp32; delta_theta is input dtype → theta is fp32 (upcast if needed)
        new_angle_state = theta
        cap('theta', theta)

        cos = torch.cos(theta)   # (B, nheads, num_rope_angles), fp32
        sin = torch.sin(theta)

        # Expand for (B, 1, nheads, num_rope_angles) broadcasting on K, Q
        cos_4d = cos.unsqueeze(1)
        sin_4d = sin.unsqueeze(1)

        # Apply pairwise rotation [§3 iii]
        K_rot = rope_pairwise(K_pre_rope, cos_4d, sin_4d, self.rotary_dim)
        Q_rot = rope_pairwise(Q_pre_rope, cos_4d, sin_4d, self.rotary_dim)
        cap('K_rot', K_rot)
        cap('Q_rot', Q_rot)

        # Reshape x and z: (B, d_inner) → (B, nheads, headdim)
        x = x_raw.reshape(B, self.nheads, self.headdim)   # V at time t
        z = z_raw.reshape(B, self.nheads, self.headdim)

        # ── Stage 5: discretisation coefficients  [§4] ──────────────────────
        alpha = torch.exp(A * DT)             # (B, nheads)
        beta  = (1.0 - lam) * DT * alpha     # (B, nheads) — weight of past
        gamma = lam * DT                      # (B, nheads) — weight of present

        cap('alpha', alpha)
        cap('beta',  beta)
        cap('gamma', gamma)

        # ── Stage 6: SSM recurrence  [§4] ───────────────────────────────────
        # Cast all SSM inputs to ssm_state.dtype (always fp32) to avoid mixed-dtype
        # einsum errors and to match the original Triton kernel which accumulates in fp32.
        # Using .to(ssm_state.dtype) rather than .float() so the cast is relative, not
        # hardcoded to torch.float32.
        sd = ssm_state.dtype                             # fp32 always (see allocate_states)
        K      = K_rot.squeeze(1).to(sd)               # (B, nheads, d_state)
        Q      = Q_rot.squeeze(1).to(sd)               # (B, nheads, d_state)
        k_prev = k_state.squeeze(1).to(sd)             # (B, nheads, d_state)
        x_s    = x.to(sd)                              # (B, nheads, headdim)
        v_s    = v_state.to(sd)                        # (B, nheads, headdim)

        # Outer products: V ⊗ K → (B, nheads, headdim, d_state)  [§4 Δh formula]
        outer_curr = torch.einsum("bnh,bns->bnhs", x_s,  K)
        outer_prev = torch.einsum("bnh,bns->bnhs", v_s,  k_prev)

        # Δh = β·(V_{t-1}⊗K_{t-1}) + γ·(V_t⊗K_t)
        g4 = gamma.to(sd)[:, :, None, None]
        b4 = beta.to(sd)[ :, :, None, None]
        delta_h = b4 * outer_prev + g4 * outer_curr
        cap('delta_h', delta_h)
        cap('ssm_state_before', ssm_state)

        # h = α·h_{t-1} + Δh
        new_ssm_state = alpha.to(sd)[:, :, None, None] * ssm_state + delta_h
        cap('ssm_state_after', new_ssm_state)

        # State updates — cast K back to the original k_state dtype, mirroring the Triton
        # kernel which stores fp32 rotated K to the fp16 output tensor.
        new_k_state = K_rot.to(k_state.dtype)   # (B, 1, nheads, d_state)
        new_v_state = x                          # (B, nheads, headdim)

        # ── Stage 7: output y  [§5] ─────────────────────────────────────────
        # y = h·Q  — both in sd (fp32), result is fp32
        y = torch.einsum("bnhs,bns->bnh", new_ssm_state, Q)   # (B, nheads, headdim)
        cap('y_after_hQ', y)

        # y = y + D·V  (skip connection)
        y = y + self.D.to(sd)[None, :, None] * x_s
        cap('y_after_DV', y)

        # y = y ⊙ SiLU(z)  (gate)
        y = y * F.silu(z.to(sd))
        cap('y_after_gate', y)

        # ── Stage 8: output projection  [§6] ────────────────────────────────
        y_flat = y.reshape(B, self.d_inner)
        # Cast to weight dtype to handle fp32 y in fp16 model (from SSM fp32 promotion)
        out = self.out_proj(y_flat.to(self.out_proj.weight.dtype))
        cap('out', out)

        return out, new_angle_state, new_ssm_state, new_k_state, new_v_state

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def step(
        self,
        u: torch.Tensor,
        angle_state: torch.Tensor,
        ssm_state: torch.Tensor,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        **kwargs,
    ):
        """
        Single autoregressive step — same public signature as Mamba3.step().

        Args:
            u:           (B, d_model)
            angle_state: (B, nheads, num_rope_angles)
            ssm_state:   (B, nheads, headdim, d_state)
            k_state:     (B, 1, nheads, d_state)
            v_state:     (B, nheads, headdim)
        Returns:
            out, new_angle_state, new_ssm_state, new_k_state, new_v_state
        """
        return self._compute_step(u, angle_state, ssm_state, k_state, v_state)

    def forward(
        self,
        u_seq: torch.Tensor,
        angle_state: torch.Tensor,
        ssm_state: torch.Tensor,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
    ):
        """
        Prefill via unrolled sequential step() — numerically identical to calling
        step() L times manually.

        Args:
            u_seq: (B, L, d_model)
        Returns:
            out_seq: (B, L, d_model)
            updated states (angle_state, ssm_state, k_state, v_state)
        """
        _, L, _ = u_seq.shape
        outputs = []
        for t in range(L):
            out, angle_state, ssm_state, k_state, v_state = self._compute_step(
                u_seq[:, t], angle_state, ssm_state, k_state, v_state
            )
            outputs.append(out.unsqueeze(1))
        return torch.cat(outputs, dim=1), angle_state, ssm_state, k_state, v_state


# ---------------------------------------------------------------------------
# MIXED precision variant
# ---------------------------------------------------------------------------

class Mamba3SISOPortableMixed(Mamba3SISOPortable):
    """
    MIXED precision variant: stage 5 (α/β/γ) and stage 6 (SSM update) are
    explicitly promoted to float32, while the rest of the model stays in its
    native dtype (fp16 after model.half()).

    ssm_state is kept in float32 (same as base class API contract).
    This matches the design rationale in docs/prompt.md §3.3 MIXED mode.
    """

    def _compute_step(
        self,
        u: torch.Tensor,
        angle_state: torch.Tensor,
        ssm_state: torch.Tensor,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        capture: dict = None,
    ):
        def cap(name: str, val: torch.Tensor) -> torch.Tensor:
            if capture is not None:
                capture[name] = val.detach().clone()
            return val

        B = u.shape[0]

        # Stages 1–4: identical to base class ───────────────────────────────
        zxBCdt = self.in_proj(u)
        cap('in_proj_out', zxBCdt)

        z_raw, x_raw, B_raw, C_raw, dd_dt, dd_A, trap_raw, angles_raw = \
            torch.split(zxBCdt, self._split, dim=-1)
        cap('z_split', z_raw); cap('x_split', x_raw)
        cap('B_split', B_raw); cap('C_split', C_raw)
        cap('dd_dt_split', dd_dt); cap('dd_A_split', dd_A)
        cap('trap_split', trap_raw); cap('angles_split', angles_raw)

        A   = -torch.clamp(F.softplus(dd_A), min=self.A_floor)
        DT  = F.softplus(dd_dt + self.dt_bias)
        lam = torch.sigmoid(trap_raw)
        cap('A_postclamp', A); cap('DT', DT); cap('lambda', lam)

        Kn = self.B_norm(B_raw.reshape(B, 1, 1, self.d_state))
        Qn = self.C_norm(C_raw.reshape(B, 1, 1, self.d_state))
        cap('B_norm_out', Kn); cap('C_norm_out', Qn)

        Kn = Kn.expand(-1, -1, self.nheads, -1)
        Qn = Qn.expand(-1, -1, self.nheads, -1)
        cap('B_bcast', Kn); cap('C_bcast', Qn)

        K_bias = self.B_bias.transpose(0, 1)
        Q_bias = self.C_bias.transpose(0, 1)
        K_pre_rope = Kn + K_bias[None]
        Q_pre_rope = Qn + Q_bias[None]
        cap('K_pre_rope', K_pre_rope); cap('Q_pre_rope', Q_pre_rope)

        angles = angles_raw.unsqueeze(1).expand(-1, self.nheads, -1)
        delta_theta = torch.tanh(angles) * torch.pi * DT.unsqueeze(-1)
        cap('delta_theta', delta_theta)

        # DEVIATION: no mod 2π (same as base class)
        theta = angle_state + delta_theta
        new_angle_state = theta
        cap('theta', theta)

        cos, sin = torch.cos(theta), torch.sin(theta)
        cos_4d, sin_4d = cos.unsqueeze(1), sin.unsqueeze(1)

        K_rot = rope_pairwise(K_pre_rope, cos_4d, sin_4d, self.rotary_dim)
        Q_rot = rope_pairwise(Q_pre_rope, cos_4d, sin_4d, self.rotary_dim)
        cap('K_rot', K_rot); cap('Q_rot', Q_rot)

        x = x_raw.reshape(B, self.nheads, self.headdim)
        z = z_raw.reshape(B, self.nheads, self.headdim)

        # Stage 5: α/β/γ explicitly in fp32 ─────────────────────────────────
        A_32  = A.float()
        DT_32 = DT.float()
        lam_32 = lam.float()

        alpha = torch.exp(A_32 * DT_32)
        beta  = (1.0 - lam_32) * DT_32 * alpha
        gamma = lam_32 * DT_32
        cap('alpha', alpha); cap('beta', beta); cap('gamma', gamma)

        # Stage 6: SSM update explicitly in fp32 ─────────────────────────────
        K = K_rot.squeeze(1).float()
        Q = Q_rot.squeeze(1).float()
        k_prev = k_state.squeeze(1).float()
        x_32   = x.float()
        v_32   = v_state.float()

        outer_curr = torch.einsum("bnh,bns->bnhs", x_32,  K)
        outer_prev = torch.einsum("bnh,bns->bnhs", v_32,  k_prev)

        g4 = gamma[:, :, None, None]
        b4 = beta[:,  :, None, None]
        delta_h = b4 * outer_prev + g4 * outer_curr
        cap('delta_h', delta_h)
        cap('ssm_state_before', ssm_state)

        new_ssm_state = alpha[:, :, None, None] * ssm_state + delta_h
        cap('ssm_state_after', new_ssm_state)

        new_k_state = K_rot.to(k_state.dtype)   # cast back to state dtype (mirrors Triton kernel)
        new_v_state = x

        # Stages 7–8: same as base class ─────────────────────────────────────
        y = torch.einsum("bnhs,bns->bnh", new_ssm_state, Q)
        cap('y_after_hQ', y)
        y = y + self.D[None, :, None].float() * x_32
        cap('y_after_DV', y)
        y = y * F.silu(z.float())
        cap('y_after_gate', y)

        y_flat = y.reshape(B, self.d_inner)
        out = self.out_proj(y_flat.to(self.out_proj.weight.dtype))
        cap('out', out)

        return out, new_angle_state, new_ssm_state, new_k_state, new_v_state
