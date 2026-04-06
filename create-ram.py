import os

dirs = [
    "relation_aware",
    "examples",
    "tests"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Created directory: {d}")

files = {
    "relation_aware/__init__.py": """from .module import RelationAwareModule

__version__ = "1.0.0"
__author__ = "Han Dajing"
__all__ = ["RelationAwareModule"]
""",
    "relation_aware/module.py": """import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationAwareModule(nn.Module):
    def __init__(self, in_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.dropout = dropout

        assert self.head_dim * num_heads == in_dim, \
            f"in_dim ({in_dim}) not divisible by num_heads ({num_heads})"

        self.qkv_proj = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.out_proj = nn.Linear(in_dim, in_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        residual = x
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(attn_out)
        return self.norm(residual + self.drop(out))
""",
}

for file_path, file_content in files.items():
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(file_content)
    print(f"Generated: {file_path}")

print("\n" + "=" * 50)
print("Relation-Aware Module project generated successfully!")
print("=" * 50)
print("\nNext steps:")
print("1. cd relation-aware-module")
print("2. pip install -r requirements.txt")
print("3. pip install -e .")
print("4. python examples/demo.py")
