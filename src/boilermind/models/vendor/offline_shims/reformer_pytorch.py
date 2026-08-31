"""Offline shim for the missing `reformer_pytorch` package.

Time-Series-Library's layers/SelfAttention_Family.py imports
`from reformer_pytorch import LSHSelfAttention` at module top, but that
package is only needed by the Reformer attention classes. PatchTST uses
FullAttention and never touches it. This shim makes the import succeed and
raises loudly if the Reformer path is actually used.
"""
import torch


class LSHSelfAttention(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            'offline shim: reformer_pytorch is not installed. '
            'Reformer attention (LSHSelfAttention) is unavailable; '
            'this does not affect PatchTST.')
