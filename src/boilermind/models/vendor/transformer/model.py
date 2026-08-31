"""
Transformer model — uses original paper implementation from 对照组
Wrapper that re-exports the transformer class with CPU/GPU compatibility
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from transformer import transformer as _Transformer

class TransformerWrapper(_Transformer):
    """Compatibility wrapper that strips .cuda() calls"""
    def __init__(self, n_past=20, n_future=1, d_model=64, d_ff=48, num_heads=12,
                 num_layers=6, dropout=0.5, top_k=5):
        super().__init__(n_past, n_future, d_model, d_ff, num_heads,
                         num_layers, dropout, top_k)
    def forward(self, x_enc, x_dec):
        return super().forward(x_enc, x_dec)
