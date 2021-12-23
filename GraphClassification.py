from types import new_class
import dgl.nn.pytorch as dglnn
from numpy.lib.type_check import imag
import torch.nn as nn
import torch
import dgl
from torch.utils import data
from Graph_dataloader import PowerGNNdataset
from dgl.dataloading import GraphDataLoader
import torch.nn.functional as F
from dgl.data import register_data_args
import argparse
import time
import numpy as np


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()

        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg)



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
    n_classes = dataset.n_classes

    model = Classifier(in_feats,
                      args.n_hidden,
                      n_classes,
                    ).to(torch.double)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()

        for batched_graph, labels in dataloader:
            feats = batched_graph.ndata['electric']
            if cuda:
                labels = labels.cuda()
                feats = feats.cuda()
                batched_graph = batched_graph.to("cuda:0")
               
            # forward
            logits = model(batched_graph, feats)
            loss = loss_fcn(logits, labels)

            if epoch >= 3:
                dur.append(time.time() - t0)

                _, indices = torch.max(logits, dim=1)
                correct = torch.sum(indices == labels)
                acc = correct.item() * 1.0 / labels.shape[0]
                print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                            .format(epoch, np.mean(dur), loss.item(), acc,))

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
    parser.add_argument("--lr", type=float, default=1e-0,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    # parser.add_argument("--n-layers", type=int, default=1,
    #                     help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    # parser.add_argument("--aggregator-type", type=str, default="pool",
    #                     help="Aggregator type: mean/gcn/pool/lstm")
    args = parser.parse_args()
    print(args)

    main(args)