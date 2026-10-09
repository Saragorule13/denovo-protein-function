"""GCN / GAT / ProteinGT (TransformerConv) graph classifiers, shared head design.

Common: in-proj 509->H, L message layers (residual + LayerNorm + ReLU + dropout),
mean||max pool -> MLP(2H->H->N).
"""
import torch
import torch.nn as nn
from torch_geometric.nn import (GATConv, GCNConv, TransformerConv,
                                global_max_pool, global_mean_pool)


class Head(nn.Module):
    def __init__(self, h, n_cls, drop=0.2):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(h, n_cls))

    def forward(self, x, batch):
        return self.mlp(torch.cat([global_mean_pool(x, batch),
                                   global_max_pool(x, batch)], dim=1))


class _Base(nn.Module):
    def __init__(self, h=256, n_cls=65, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(509, h)
        self.norm0 = nn.LayerNorm(h)
        self.drop = nn.Dropout(drop)
        self.head = Head(h, n_cls, drop=0.2)

    def block(self, h, conv_out):
        return nn.Sequential(nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.1),
                             conv_out)[-3:] if False else (h, conv_out)


class GCNNet(_Base):
    def __init__(self, h=256, layers=3, n_cls=65, drop=0.1):
        super().__init__(h, n_cls, drop)
        self.convs = nn.ModuleList([GCNConv(h, h) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(layers)])

    def forward(self, x, edge_index, batch, edge_weight=None, edge_attr=None):
        x = self.drop(torch.relu(self.norm0(self.proj(x))))
        for conv, norm in zip(self.convs, self.norms):
            x = x + self.drop(conv(norm(x), edge_index, edge_weight))
            x = torch.relu(x)
        return self.head(x, batch)


class GATNet(_Base):
    def __init__(self, h=256, layers=3, heads=4, n_cls=65, drop=0.1):
        super().__init__(h, n_cls, drop)
        assert h % heads == 0
        self.convs = nn.ModuleList(
            [GATConv(h, h // heads, heads=heads, edge_dim=19) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(layers)])

    def forward(self, x, edge_index, batch, edge_weight=None, edge_attr=None):
        x = self.drop(torch.relu(self.norm0(self.proj(x))))
        for conv, norm in zip(self.convs, self.norms):
            x = x + self.drop(conv(norm(x), edge_index, edge_attr))
            x = torch.relu(x)
        return self.head(x, batch)


class ProteinGT(_Base):
    """Graph Transformer: TransformerConv x L, edge-aware attention, residuals."""

    def __init__(self, h=256, layers=4, heads=4, n_cls=65, drop=0.1):
        super().__init__(h, n_cls, drop)
        assert h % heads == 0
        self.convs = nn.ModuleList(
            [TransformerConv(h, h // heads, heads=heads, edge_dim=19,
                             dropout=drop) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(layers)])

    def forward(self, x, edge_index, batch, edge_weight=None, edge_attr=None,
                return_attn=False):
        x = self.drop(torch.relu(self.norm0(self.proj(x))))
        attns = []
        for conv, norm in zip(self.convs, self.norms):
            if return_attn:
                out, (ei, aw) = conv(norm(x), edge_index, edge_attr,
                                     return_attention_weights=True)
                attns.append((ei, aw.mean(dim=1)))  # mean over heads
                x = x + self.drop(out)
            else:
                x = x + self.drop(conv(norm(x), edge_index, edge_attr))
            x = torch.relu(x)
        logits = self.head(x, batch)
        return (logits, attns) if return_attn else logits


def make(name, **kw):
    return {"gcn": GCNNet, "gat": GATNet, "gt": ProteinGT}[name](**kw)