from matplotlib.pyplot import axes
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def to_cuda(x):
    return x.cuda()

class GATLayerAdj(nn.Module):
    """
    More didatic (also memory-hungry) GAT layer
    """

    def __init__(self,d_i,d_o,act=F.relu,eps=1e-6):
        super(GATLayerAdj,self).__init__()
        self.f = nn.Linear(2*d_i,d_o)
        self.w = nn.Linear(2*d_i,1)
        self.act = act
        self._init_weights()
        
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)
    
    def forward(self,x,adj,src,tgt,Msrc,Mtgt):
        """
        features -> N,i node features
        adj -> N,N adjacency matrix
        src -> E,i source index for edges
        tgt -> E,i target index for edges
        Msrc -> N,E adjacency matrix from source nodes to edges
        Mtgt -> N,E adjacency matrix from target nodes to edges
        """
        N = x.size()[0]
        hsrc = x.unsqueeze(0).expand(N,-1,-1) # 1,N,i
        htgt = x.unsqueeze(1).expand(-1,N,-1) # N,1,i
        
        h = torch.cat([hsrc,htgt],dim=2) # N,N,2i
        
        a = self.w(h) # N,N,1
        a_sqz = a.squeeze(2) # N,N
        a_zro = -1e16*torch.ones_like(a_sqz) # N,N
        a_msk = torch.where(adj>0,a_sqz,a_zro) # N,N
        a_att = F.softmax(a_msk,dim=1) # N,N
        
        y = self.act(self.f(h)) # N,N,o
        y_att = a_att.unsqueeze(-1)*y # N,N,o
        o = y_att.sum(dim=1).squeeze()
        
        return o

class GATLayerEdgeAverage(nn.Module):
    """
    GAT layer with average, instead of softmax, attention distribution
    """
    def __init__(self,d_i,d_o,act=F.relu,eps=1e-6):
        super(GATLayerEdgeAverage,self).__init__()
        self.f = nn.Linear(2*d_i,d_o)
        self.w = nn.Linear(2*d_i,1)
        self.act = act
        self._init_weights()
        self.eps = eps
        
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)
    
    def forward(self,x,adj,src,tgt,Msrc,Mtgt):
        """
        features -> N,i node features
        adj -> N,N adjacency matrix
        src -> E,i source index for edges
        tgt -> E,i target index for edges
        Msrc -> N,E adjacency matrix from source nodes to edges
        Mtgt -> N,E adjacency matrix from target nodes to edges
        """
        hsrc = x[src] # E,i
        htgt = x[tgt] # E,i
        h = torch.cat([hsrc,htgt],dim=1) # E,2i
        y = self.act(self.f(h)) # E,o
        a = self.w(h) # E,1
        a_sum = torch.mm(Mtgt,a) + self.eps # N,E x E,1 = N,1
        o = torch.mm(Mtgt,y * a) / a_sum # N,1
        assert not torch.isnan(o).any()

        return o

class GATLayerEdgeSoftmax(nn.Module):
    """
    GAT layer with softmax attention distribution (May be prone to numerical errors)
    """
    def __init__(self,d_i,d_o,act=F.relu,eps=1e-6):
        super(GATLayerEdgeSoftmax,self).__init__()
        self.f = nn.Linear(2*d_i,d_o)
        self.w = nn.Linear(2*d_i,1)
        self.act = act
        self._init_weights()
        self.eps = eps
        
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)
    
    def forward(self,x,adj,src,tgt,Msrc,Mtgt):
        """
        features -> N,i node features
        adj -> N,N adjacency matrix
        src -> E,i source index for edges
        tgt -> E,i target index for edges
        Msrc -> N,E adjacency matrix from source nodes to edges
        Mtgt -> N,E adjacency matrix from target nodes to edges
        """
        hsrc = x[src] # E,i
        htgt = x[tgt] # E,i
        h = torch.cat([hsrc,htgt],dim=1) # E,2i
        y = self.act(self.f(h)) # E,o
        # FIXME Manual softmax doesn't as expected numerically
        a = self.w(h) # E,1
        assert not torch.isnan(a).any()
        a_base, _ = torch.max(a,0,keepdim=True)#[0] + self.eps
        assert not torch.isnan(a_base).any()
        a_norm = a-a_base
        assert not torch.isnan(a_norm).any()
        a_exp = torch.exp(a_norm)
        assert not torch.isnan(a_exp).any()
        a_sum = torch.mm(Mtgt,a_exp) + self.eps # N,E x E,1 = N,1
        assert not torch.isnan(a_sum).any()
        o = torch.mm(Mtgt,y * a_exp) / a_sum # N,1
        assert not torch.isnan(o).any()

        return o


class GATLayerMultiHead(nn.Module):
    
    def __init__(self,d_in,d_out,num_heads):
        super(GATLayerMultiHead,self).__init__()
        
        self.GAT_heads = nn.ModuleList(
              [
                GATLayerEdgeSoftmax(d_in,d_out)
                for _ in range(num_heads)
              ]
        )
    
    def forward(self,x,adj,src,tgt,Msrc,Mtgt):
        return torch.cat([l(x,adj,src,tgt,Msrc,Mtgt) for l in self.GAT_heads],dim=1)


class GAT_MNIST(nn.Module):
    
    def __init__(self,num_features,num_classes,num_heads=[2,2,2]):
        super(GAT_MNIST,self).__init__()
        
        self.layer_heads = [1]+num_heads
        self.GAT_layer_sizes = [num_features,32,64,64]
        
        self.MLP_layer_sizes = [self.layer_heads[-1]*self.GAT_layer_sizes[-1],32,num_classes]
        self.MLP_acts = [F.relu,lambda x:x]
        
        self.GAT_layers = nn.ModuleList(
              [
                GATLayerMultiHead(d_in*heads_in,d_out,heads_out)
                for d_in,d_out,heads_in,heads_out in zip(
                    self.GAT_layer_sizes[:-1],
                    self.GAT_layer_sizes[1:],
                    self.layer_heads[:-1],
                    self.layer_heads[1:],
                )
              ]
        )
        self.MLP_layers = nn.ModuleList(
              [
                nn.Linear(d_in,d_out)
                for d_in,d_out in zip(self.MLP_layer_sizes[:-1],self.MLP_layer_sizes[1:])
              ]
        )
    
    def forward(self,x,adj,src,tgt,Msrc,Mtgt,Mgraph):
        for l in self.GAT_layers:
            x = l(x,adj,src,tgt,Msrc,Mtgt)
        x = torch.mm(Mgraph.t(),x)
        for layer,act in zip(self.MLP_layers,self.MLP_acts):
            x = act(layer(x))
        return x
        

class GAT_MNIST_20191016134(nn.Module):
    
    def __init__(self,num_features,num_classes):
        super(GAT_MNIST,self).__init__()
        
        self.GAT_layer_sizes = [num_features,32,64,64]
        self.MLP_layer_sizes = [self.GAT_layer_sizes[-1],32,num_classes]
        self.MLP_acts = [F.relu,lambda x:x]
        
        self.GAT_layers = nn.ModuleList(
              [
                GATLayerEdgeSoftmax(d_in,d_out)
                for d_in,d_out in zip(self.GAT_layer_sizes[:-1],self.GAT_layer_sizes[1:])
              ]
        )
        self.MLP_layers = nn.ModuleList(
              [
                nn.Linear(d_in,d_out)
                for d_in,d_out in zip(self.MLP_layer_sizes[:-1],self.MLP_layer_sizes[1:])
              ]
        )
    
    def forward(self,x,adj,src,tgt,Msrc,Mtgt,Mgraph):
        for l in self.GAT_layers:
            x = l(x,adj,src,tgt,Msrc,Mtgt)
        x = torch.mm(Mgraph.t(),x)
        for layer,act in zip(self.MLP_layers,self.MLP_acts):
            x = act(layer(x))
        return x

class CGAT(nn.Module): 

    def __init__(self,num_features,num_classes,num_heads=[2,2,2]):
        super(CGAT,self).__init__()
        
        
        self.CONV_layer_sizes = [num_features, 32, 32]

        self.layer_heads = [1]+num_heads
        self.GAT_layer_sizes = [self.CONV_layer_sizes[-1],32,64,64]
        
        self.MLP_layer_sizes = [self.layer_heads[-1]*self.GAT_layer_sizes[-1],32,num_classes]
        self.MLP_acts = [F.relu,lambda x:x]

        self.CONV_layers = nn.ModuleList(
              [
                nn.Sequential(
                    nn.Conv2d(d_in, d_out, kernel_size=3),
                    nn.BatchNorm2d(d_out),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                )
                for d_in,d_out in zip(self.CONV_layer_sizes[:-1], self.CONV_layer_sizes[1:])
              ]
        ) # adding convolutional layers
        
        self.GAT_layers = nn.ModuleList(
              [
                GATLayerMultiHead(d_in*heads_in,d_out,heads_out)
                for d_in,d_out,heads_in,heads_out in zip(
                    self.GAT_layer_sizes[:-1],
                    self.GAT_layer_sizes[1:],
                    self.layer_heads[:-1],
                    self.layer_heads[1:],
                )
              ]
        )
        self.MLP_layers = nn.ModuleList(
              [
                nn.Linear(d_in,d_out)
                for d_in,d_out in zip(self.MLP_layer_sizes[:-1],self.MLP_layer_sizes[1:])
              ]
        )

    def build_graph(self, x): # x = [b, c, w, h]

        b, c, w, h = x.shape[:]
       
        n = w * h 
        edges = []
        for i in range(0, w):
            for j in range(0, h):
                edges.append([i*(w - 1) + j, i*(w -1) + j]) #self loop
                edges.append([i*(w - 1) + j, i*(w - 1) + j + 1]) #right vertex
                edges.append([i*(w - 1) + j, i*(w) + j]) # below vertex
        
        edges = np.array(edges, dtype=np.int64)       
        edges = np.tile(edges, (b, 1)).reshape(b, -1, 2)
        x = x.reshape(-1, c, w * h).permute((0, 2, 1)) #reshape of node features in a graph, 
        x = x.reshape(-1, c) #batched graph --> concatenate all nodes 
        
        dummy_nodes = np.random.randn(b, w*h, c) #create a dummy node for make adjency matrix
        gs = [(h, e) for (h, e) in zip(dummy_nodes, edges)]
        
        N = sum(g[0].shape[0] for g in gs)
        M = sum(g[1].shape[0] for g in gs)
        adj = np.zeros([N,N])
        src = np.zeros([M])
        tgt = np.zeros([M])
        Msrc = np.zeros([N,M])
        Mtgt = np.zeros([N,M])
        Mgraph = np.zeros([N,b])
        h = np.concatenate([g[0] for g in gs])
        
        n_acc = 0
        m_acc = 0
        for g_idx, g in enumerate(gs):
            n = g[0].shape[0]
            m = g[1].shape[0]
            
            for e,(s,t) in enumerate(g[1]):
                adj[n_acc+s,n_acc+t] = 1
                adj[n_acc+t,n_acc+s] = 1
                
                src[m_acc+e] = n_acc+s
                tgt[m_acc+e] = n_acc+t
                
                Msrc[n_acc+s,m_acc+e] = 1
                Mtgt[n_acc+t,m_acc+e] = 1
                
            Mgraph[n_acc:n_acc+n,g_idx] = 1
            
            n_acc += n
            m_acc += m
        #endfor

        adj,src,tgt,Msrc,Mtgt,Mgraph = map(torch.from_numpy,(
                            adj.astype(np.float32),
                            src.astype(np.int64),
                            tgt.astype(np.int64),
                            Msrc.astype(np.float32),
                            Mtgt.astype(np.float32),
                            Mgraph.astype(np.float32),))
        if (x.is_cuda):
            adj,src,tgt,Msrc,Mtgt,Mgraph = map(to_cuda,(adj,src,tgt,Msrc,Mtgt,Mgraph))
        return (
            x, adj, src, tgt, Msrc, Mtgt, Mgraph
        )

    def forward(self, x):

        for l in self.CONV_layers: # adding convolutional layers
            x = l(x)
        
        x, adj, src, tgt, Msrc, Mtgt, Mgraph = self.build_graph(x)

        for l in self.GAT_layers:
            x = l(x,adj,src,tgt,Msrc,Mtgt)
        x = torch.mm(Mgraph.t(),x)
        for layer,act in zip(self.MLP_layers,self.MLP_acts):
            x = act(layer(x))
        return x


if __name__ == "__main__":
    g = CGAT(num_features=3, num_classes=10).cuda()
    # x = torch.rand(2, 3, 2, 2)
    # b = CGAT.build_graph(x)
    images = torch.rand(2, 3, 32, 32).cuda()
    y = g(images)
    print(y.shape)
    
