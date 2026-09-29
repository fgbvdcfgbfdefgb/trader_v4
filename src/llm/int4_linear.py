"""
Pure-PyTorch group-wise INT4 linear layer.

Depends on torch only -- no bitsandbytes, no llama.cpp, no AWQ/GPTQ runtime.
That is the whole point: the training host cannot install packages.

Storage (produced by scripts/quantize_mistral_int4.py):
    qweight uint8   [out, in//2]   two 4-bit codes per byte
                                   low nibble  -> even input index
                                   high nibble -> odd  input index
    scale   float16 [out, in//G]
    zero    float16 [out, in//G]
    W ~= q.to(dtype) * scale + zero

Two execution modes:
    materialize=False  dequantise inside forward().  ~3.9 GB resident for a
                       7B model, but every forward pays the unpack cost.
    materialize=True   dequantise once and keep the dense weight.  Fast, but
                       needs ~14 GB (bf16) / 28 GB (fp32) for a 7B model.

Batching matters a lot in the non-materialised mode: the unpack cost is paid
per *forward call*, not per row, so running 8 prompts at once is close to 8x
cheaper per prompt.  The advisor daemon exploits this.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Int4Linear", "dequantize_int4"]


def dequantize_int4(qweight: torch.Tensor, scale: torch.Tensor,
                    zero: torch.Tensor, group: int,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """qweight[out, in//2] uint8 -> dense [out, in] tensor of `dtype`."""
    out_f, half = qweight.shape
    lo = qweight & 0x0F
    hi = qweight >> 4
    # interleave so column order is  lo0, hi0, lo1, hi1, ...
    q = torch.stack((lo, hi), dim=2).reshape(out_f, half * 2)
    q = q.reshape(out_f, -1, group).to(dtype)
    w = q * scale.unsqueeze(-1).to(dtype) + zero.unsqueeze(-1).to(dtype)
    return w.reshape(out_f, half * 2)


class Int4Linear(nn.Module):
    """nn.Linear replacement backed by 4-bit weights."""

    def __init__(self, in_features: int, out_features: int, group: int = 128,
                 bias: bool = False, materialize: bool = False,
                 compute_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group = group
        self.materialize = materialize
        self.compute_dtype = compute_dtype

        self.register_buffer(
            "qweight", torch.zeros((out_features, in_features // 2),
                                   dtype=torch.uint8), persistent=False)
        self.register_buffer(
            "scale", torch.zeros((out_features, in_features // group),
                                 dtype=torch.float16), persistent=False)
        self.register_buffer(
            "zero", torch.zeros((out_features, in_features // group),
                                dtype=torch.float16), persistent=False)
        self.register_buffer("bias", None, persistent=False)
        self._dense: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    def load(self, qweight: torch.Tensor, scale: torch.Tensor,
             zero: torch.Tensor) -> None:
        self.qweight = qweight
        self.scale = scale
        self.zero = zero
        if self.materialize:
            self._dense = dequantize_int4(
                qweight, scale, zero, self.group, self.compute_dtype)
            # free the packed copies
            self.qweight = torch.empty(0, dtype=torch.uint8)
            self.scale = torch.empty(0, dtype=torch.float16)
            self.zero = torch.empty(0, dtype=torch.float16)

    def dense(self) -> torch.Tensor:
        if self._dense is not None:
            return self._dense
        return dequantize_int4(self.qweight, self.scale, self.zero,
                               self.group, self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dense()
        if x.dtype != w.dtype:
            x = x.to(w.dtype)
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"group={self.group}, 4bit, "
                f"materialized={self._dense is not None}")

    @property
    def nbytes(self) -> int:
        if self._dense is not None:
            return self._dense.numel() * self._dense.element_size()
        return (self.qweight.numel() + 2 * 2 * self.scale.numel())
