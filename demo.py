import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relation_aware import RelationAwareModule

def main():
    print("=== Relation-Aware Module (RAM) Demo ===")
    print("Based on Anti-Entropic Life Theory\n")
    
    in_dim, num_heads = 256, 8
    model = RelationAwareModule(in_dim=in_dim, num_heads=num_heads)
    model.eval()
    
    x = torch.randn(2, 10, in_dim)
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print("\nModule loaded successfully! Demo passed.")

if __name__ == "__main__":
    main()
