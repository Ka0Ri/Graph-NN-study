import dgl
import torch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import re
import pandas as pd
import networkx as nx
import numpy as np
from sklearn import preprocessing


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
        data, label = read_all_data(xls_data)
        self.graphs = []
        for graph in data: 
            n = len(graph)
            nx_g = nx.path_graph(n)
            g = dgl.from_networkx(nx_g)
            g.ndata['electric'] = torch.tensor(np.expand_dims(np.array(graph), axis=-1))
            self.graphs.append(g)
        le = preprocessing.LabelEncoder()
        le.fit(label)
        self.label = le.transform(label)
        

    def __getitem__(self, idx):
        # assert idx == 0, "This dataset has only one graph"
        return self.graphs[idx], self.label[idx]

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