"""LRP-modified backward passes for r-lens training.

The r-lens (https://www.alignmentforum.org/posts/nv8oedrnLXKRzNEL9) is fit
exactly like the J-lens, but with three Layer-wise Relevance Propagation rules
applied to the backward pass. Each rule is implemented by rewriting a module's
forward so that the value is unchanged but autograd sees a modified graph:

- LN-rule (residual-stream RMSNorms): treat the normalization denominator as a
  constant, making the norm linear and preventing relevance collapse.
- Identity-rule (SiLU in gated MLPs): detach the nonlinear factor of SiLU, so
  the activation's backward becomes a per-element linear map.
- Half-rule (gated MLP product): split relevance evenly across the gate's two
  branches instead of double-counting through the product.

Per the post (dense-model recipe): linear layers, attention, and q/k norms are
left unmodified. Only residual-stream RMSNorms and the gated MLPs are patched.
"""
import types

import torch
from torch import nn

# Attribute names under which residual-stream RMSNorms live on a decoder layer.
RESIDUAL_NORM_NAMES = ("input_layernorm", "post_attention_layernorm",
                      "pre_feedforward_layernorm", "post_feedforward_layernorm",
                      "post_per_layer_input_norm")


def _rmsnorm_ln_rule_forward(self, x):
    """HF-standard RMSNorm with the denominator detached (LN-rule)."""
    dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(-1, keepdim=True)
    denom = torch.rsqrt(var + self.variance_epsilon).detach()
    return (x * denom).to(dtype) * self.weight


def _rmsnorm_ln_rule_norm(self, x):
    """Detached-denominator replacement for RMSNorm._norm (Qwen3.5-style
    classes, which handle weight/casting in their own forward)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps).detach()


def _gated_mlp_lrp_forward(self, x):
    """Gated MLP (down(act(gate(x)) * up(x))) with Identity- and Half-rules.

    Value-identical to the original forward; only gradients differ:
    - act(g) = g * [act(g)/g] with the nonlinear factor detached, so the
      activation's backward is a per-element linear map (Identity-rule).
      Works for any gating activation (SiLU, GELU-tanh, ...); both have
      act(g)/g -> act'(0) = 0.5 as g -> 0, used where g is (near) zero.
    - a * b -> 0.5 * (a.detach() * b + a * b.detach()) (Half-rule).
    """
    g = self.gate_proj(x)
    u = self.up_proj(x)
    act_val = self.act_fn(g)
    factor = torch.where(g.abs() > 1e-6, act_val / g,
                         torch.full_like(g, 0.5)).detach()
    # straight-through formulations: the (y - y.detach()) terms are exactly
    # zero in value, so the forward is bit-identical to the original module
    # while the backward sees the LRP-modified linearizations
    act = act_val.detach() + (g - g.detach()) * factor          # Identity-rule
    prod_val = (act_val * u).detach()
    prod = (prod_val                                            # Half-rule
            + 0.5 * act.detach() * (u - u.detach())
            + 0.5 * (act - act.detach()) * u.detach())
    return self.down_proj(prod)


def _is_gated_mlp(module: nn.Module) -> bool:
    return all(hasattr(module, a)
               for a in ("gate_proj", "up_proj", "down_proj", "act_fn"))


def apply_lrp_rules(layers: nn.ModuleList, final_norm: nn.Module | None = None) -> dict:
    """Patch decoder layers in place with the r-lens LRP rules.

    Returns a summary dict of how many modules of each kind were patched.
    """
    patched = {"rmsnorm": 0, "gated_mlp": 0}

    def patch_norm(norm):
        if hasattr(norm, "_norm") and hasattr(norm, "eps"):
            # patch only the normalization; the module's own forward still
            # applies its weight/casting convention (e.g. Qwen3.5's 1+weight)
            norm._norm = types.MethodType(_rmsnorm_ln_rule_norm, norm)
        elif hasattr(norm, "variance_epsilon"):
            norm.forward = types.MethodType(_rmsnorm_ln_rule_forward, norm)
        else:
            raise ValueError(f"{type(norm).__name__} does not look like an RMSNorm")
        patched["rmsnorm"] += 1

    for layer in layers:
        for name in RESIDUAL_NORM_NAMES:
            norm = getattr(layer, name, None)
            if norm is not None:
                patch_norm(norm)
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and _is_gated_mlp(mlp):
            mlp.forward = types.MethodType(_gated_mlp_lrp_forward, mlp)
            patched["gated_mlp"] += 1

    if final_norm is not None:
        patch_norm(final_norm)

    return patched
