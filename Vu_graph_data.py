import dgl
import torch

u, v = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 3])
g = dgl.graph((u, v))

print(g)
print(g.nodes())