"""Minimal offline einops shim for Time-Series-Library.

The machine has no network access to pip, but the TSL codebase imports
`from einops import rearrange, repeat` at module top. Only 4 patterns are
actually used in layers/SelfAttention_Family.py (in attention classes that
PatchTST does not use). This shim implements exactly those, and raises
loudly for anything else so wrong results can never be silent.
"""
import torch


def rearrange(x, pattern, **axes_lengths):
    x = torch.as_tensor(x)
    key = pattern.strip()
    if key == 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model':
        b, ts_d, seg_num, dm = x.shape
        return x.reshape(b * ts_d, seg_num, dm)
    if key == '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model':
        b = axes_lengths['b']
        s0, seg_num, dm = x.shape
        ts_d = s0 // b
        return x.reshape(b, ts_d, seg_num, dm).permute(0, 2, 1, 3).reshape(b * seg_num, ts_d, dm)
    if key == '(b seg_num) ts_d d_model -> b ts_d seg_num d_model':
        b = axes_lengths['b']
        s0, ts_d, dm = x.shape
        seg_num = s0 // b
        return x.reshape(b, seg_num, ts_d, dm).permute(0, 2, 1, 3)
    raise NotImplementedError(f'einops shim: rearrange pattern not implemented: {pattern!r}')


def repeat(x, pattern, **axes_lengths):
    x = torch.as_tensor(x)
    key = pattern.strip()
    if key == 'seg_num factor d_model -> (repeat seg_num) factor d_model':
        n = axes_lengths['repeat']
        return x.repeat(n, 1, 1)
    raise NotImplementedError(f'einops shim: repeat pattern not implemented: {pattern!r}')
