import dgl
import torch

# edges 0->1, 0->2, 0->3, 1->3
u, v = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 3])
g = dgl.graph((u, v))

print("graph", g)
print("node", g.nodes())
print("edges", g.edges())
# Edge end nodes and edge IDs
print(g.edges(form='all'))
# g = dgl.graph((u, v), num_nodes=8)
# print(g)
# # For an undirected graph, one needs to create edges for both directions
# edges = torch.tensor([2, 5, 3]), torch.tensor([3, 5, 0])
# g64 = dgl.graph(edges)
# print("g64", g64)
# bg = dgl.to_bidirected(g64)
# print("undirected graph", bg.edges(form='all'))
g = dgl.graph(([0, 0, 1, 5], [1, 2, 2, 0]))
g.ndata['x'] = torch.ones(g.num_nodes(), 3)
g.ndata['y'] = torch.zeros(g.num_nodes(), 3)
print(g)
g.edata['x'] = torch.ones(g.num_edges(), dtype=torch.int32)
print("2 features", g)

g.ndata['y'] = torch.randn(g.num_nodes(), 5)
print(g.ndata['x'])                  # get node 1's feature
print(g.edata['x'][torch.tensor([0, 3])]) # get features of edge 0 and 3

import networkx as nx
nx_g = nx.path_graph(5)
g = dgl.from_networkx(nx_g)
print("path graph", g)
cuda_g = g.to('cuda:0')
print("g device", g.device)
print("cuda device", cuda_g.device)

graph_data = {
   ('drug', 'interacts', 'drug'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
   ('drug', 'interacts', 'gene'): (torch.tensor([0, 1]), torch.tensor([2, 3])),
   ('drug', 'treats', 'disease'): (torch.tensor([1]), torch.tensor([2]))
}
g = dgl.heterograph(graph_data)
print(g)