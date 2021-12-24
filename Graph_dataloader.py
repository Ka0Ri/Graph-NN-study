import dgl
from numpy.lib.function_base import average
import torch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import re
import pandas as pd
import networkx as nx
import numpy as np
from sklearn import preprocessing
from torch._C import dtype


def read_sample(df, min_len = 60, input_length = 10, out_put_length=1, step = 1, with_label=False):
    data = []
    label = []
    next_sample = []
   
    for columns_name in df.columns:
        columns = df[columns_name][df[columns_name].notna()]
        list_data = [float(item) for item in columns.to_list()]
        if(len(list_data) < min_len):
            continue
        else:
            for i in range(0, len(list_data) - input_length - 2, step):
                data.append(list_data[i:(i+input_length)])
                next_sample.append(list_data[(i+input_length-out_put_length + 1):(i+input_length +1)])
                label.append(columns_name)
    if(with_label == True):
        return np.array(data), np.array(next_sample), label
    else:
        return np.array(data), np.array(next_sample)

def read_all_data(df, min_len=60, step=1):

    data = []
    label = []
    for columns_name in df.columns:
        columns = df[columns_name][df[columns_name].notna()]
        list_data = [float(re.sub('[.,]', "", str(item))) for item in columns.to_list()]
        if(len(list_data) < min_len):
            continue
        else:
            data.append(list_data)
            label.append(columns_name)

    return data, label

class PowerGNNdataset(DGLDataset):
    """ Template for customizing graph datasets in DGL.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """
    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(PowerGNNdataset, self).__init__(name='dataset_name',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    def download(self):
        # download raw data to local disk
        pass

    def process(self):

        # process raw data to graphs, labels, splitting masks
        # build graph
        
        # process data to a list of graphs and a list of labels
        xls_data = pd.read_excel('sample_data.xlsx')
        data, next_sample, label = read_sample(xls_data, with_label=True)
        self.graphs = []
        self.labels = []
        for graph, l in zip(data, next_sample): 
            
            n = len(graph)
            nx_g = nx.path_graph(n)
            g = dgl.from_networkx(nx_g)
            g.ndata['electric'] = torch.tensor(np.expand_dims(np.array(graph), axis=-1))
            self.graphs.append(g)
    
            self.labels.append(l)
            # if(l < 500): #small consumption
            #     self.labels.append(0)
            # elif(l >= 500 and l <= 1000):
            #     self.labels.append(1) #large consumption
            # else: #extra consumption
            #     self.labels.append(2)
        # le = preprocessing.LabelEncoder()
        # le.fit(label)
        # self.label = le.transform(label)
        self.n_classes = len(self.labels)
        self.in_nodes = len(graph)
        self.in_feats = 1
        

    def __getitem__(self, idx):
        # assert idx == 0, "This dataset has only one graph"
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        # number of data examples
        return len(self.graphs)

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass


if __name__ == "__main__":

    dataset = PowerGNNdataset()
    # create dataloaders
    dataloader = GraphDataLoader(dataset, batch_size=5, shuffle=True)

    # training
    for epoch in range(100):
        for g, labels in dataloader:
            print(g, labels)
            # your training code here
            pass