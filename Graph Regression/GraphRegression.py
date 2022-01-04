from types import new_class
import dgl.nn.pytorch as dglnn
from numpy.lib.type_check import imag
import torch.nn as nn
import torch
from torch.autograd import Variable
import dgl
from torch.utils import data
from Graph_dataloader import PowerGNNdataset
from dgl.dataloading import GraphDataLoader
import torch.nn.functional as F
from dgl.data import register_data_args
import argparse
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Regressor(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Regressor, self).__init__()

        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        # self.conv3 = dglnn.GraphConv(hidden_dim, hidden_dim)
        # self.conv4 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.regession = nn.Linear(hidden_dim, 1)

    def forward(self, g, h):
        # Apply graph convolution and activation.
      
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        # h = F.relu(self.conv3(g, h))
        # h = F.relu(self.conv4(g, h))
       
        with g.local_scope():
            g.ndata['h'] = h
          
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
        
            y = self.regession(hg)
            return y

class MLPRegressor(nn.Module):

    def __init__(self, in_dim, hidden_dim):
        super(MLPRegressor, self).__init__()

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.regession = nn.Linear(hidden_dim, 1)

    def forward(self, g, h):
        
        h = F.relu(self.linear1(h))
        h = F.relu(self.linear2(h))
        y = self.regession(h)
      
        return y

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)
      

    def forward(self, g, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).double().cuda()
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).double().cuda()
        
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out[-1]
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out

def main(args):

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        print("use cuda:", args.gpu)

    dataset = PowerGNNdataset()
    dataloader = GraphDataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    in_feats = dataset.in_feats
    # n_classes = dataset.n_classes
    n_nodes = dataset.in_nodes

    if(args.model == "LSTM"):
        model = LSTM(
                num_classes=1, 
                input_size=in_feats, 
                hidden_size=args.n_hidden, 
                num_layers=2).to(torch.double)
    elif(args.model == "GNN"):
        model = Regressor(in_feats,
                      args.n_hidden
                    ).to(torch.double)

    elif(args.model == "MLP"):
        model = MLPRegressor(n_nodes,
                      args.n_hidden
                    ).to(torch.double)
        
    print(model)

    if cuda:
        model.cuda()
    loss_mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    writer = SummaryWriter()
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()

        for batched_graph, labels in dataloader:
            feats = batched_graph.ndata['electric']
            
            if(args.model == "MLP"):
                feats = feats.view(-1, n_nodes)
            elif(args.model == "LSTM"):
                feats = feats.view(-1, n_nodes, in_feats)
            if cuda:
                labels = labels.cuda()
                feats = feats.cuda()
                batched_graph = batched_graph.to("cuda:0")
           
            # forward
            predicted = model(batched_graph, feats)
            
            loss = loss_mse(predicted, labels)

        if (epoch >= 3 and epoch % 5 == 0):
            dur.append(time.time() - t0)

               
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} |  "
                            .format(epoch, np.mean(dur), np.sqrt(loss.item())))
            writer.add_scalar('Loss/train',np.sqrt(loss.item()), epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Classification')
    register_data_args(parser)
    # parser.add_argument("--dropout", type=float, default=0.5,
    #                     help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=10,
                        help="number of hidden gcn units")
    # parser.add_argument("--n-layers", type=int, default=1,
    #                     help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    # parser.add_argument("--aggregator-type", type=str, default="pool",
    #                     help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument("--model", type=str, default="MLP",
                        help="Regression model ['MLP', 'GNN', 'LSTM']")
    args = parser.parse_args()
    print(args)

    main(args)