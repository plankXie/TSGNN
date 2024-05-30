from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings
import copy
import numpy as np
import pandas as pd
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


table_size = (7, 18)
postable = {1:[0,0],2:[0,17],
            3: [1, 0], 4: [1, 1], 5: [1, 12], 6: [1, 13], 7: [1, 14], 8: [1, 15], 9: [1, 16],10:[1,17],
            11: [2, 0], 12: [2, 1], 13: [2, 12], 14: [2, 13], 15: [2, 14], 16: [2, 15], 17: [2, 16],18:[2,17],
            19: [3, 0], 20: [3, 1], 21: [3, 2], 22: [3, 3], 23: [3, 4], 24: [3, 5], 25: [3, 6], 26: [3, 7],
            27: [3, 8], 28: [3, 9], 29: [3, 10], 30: [3, 11], 31: [3, 12], 32: [3, 13], 33: [3, 14],
            34: [3, 15], 35: [3, 16],36:[3,17],
            37: [4, 0], 38: [4, 1], 39: [4, 2], 40: [4, 3], 41: [4, 4], 42: [4, 5], 43: [4, 6], 44: [4, 7],
            45: [4, 8], 46: [4, 9], 47: [4, 10], 48: [4, 11], 49: [4, 12], 50: [4, 13], 51: [4, 14],
            52: [4, 15], 53: [4, 16],54:[4,17],
            55: [5, 0], 56: [5, 1],57:[5,2],58:[5,2],59:[5,2],60:[5,2],61:[5,2],62:[5,2],63:[5,2],64:[5,2],65:[5,2],66:[5,2],67:[5,2],68:[5,2],69:[5,2],70:[5,2],71:[5,2], 72: [5, 3], 73: [5, 4], 74: [5, 5], 75: [5, 6], 76: [5, 7], 77: [5, 8],
            78: [5, 9], 79: [5, 10], 80: [5, 11], 81: [5, 12], 82: [5, 13], 83: [5, 14],84:[5,15],85:[5,16],86:[5,17],
            87:[6,0],88:[6,1],89:[6,2],90:[6,2],91:[6,2],92:[6,2],93:[6,2],94:[6,2],95:[6,2],96:[6,2],97:[6,2],98:[6,2],99:[6,2],100:[6,2]}

tconst = np.zeros(table_size, dtype=np.float32)
for key in postable:
    posx = postable[key][0]
    posy = postable[key][1]
    tconst[posx, posy] = -1/75

def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    if kwargs['train_size'] is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training data.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader

class GGNNInput(object):
    def __init__(self, nodes, edge_sources, edge_targets, graph_indices, node_counts):
        self.nodes = torch.Tensor(nodes)
        self.edge_sources = torch.LongTensor(edge_sources)
        self.edge_targets = torch.LongTensor(edge_targets)
        self.graph_indices = torch.LongTensor(graph_indices)
        self.node_counts = torch.Tensor(node_counts)

    def __len__(self):
        return self.nodes.size(0)

    def to(self, device):
        self.nodes = self.nodes.to(device)
        self.edge_sources = self.edge_sources.to(device)
        self.edge_targets = self.edge_targets.to(device)
        self.graph_indices = self.graph_indices.to(device)
        self.node_counts = self.node_counts.to(device)

        return self

def collate_pool(dataset_list):
    nodes = []
    edge_sources = []
    edge_targets = []
    graph_indices = []
    node_counts = []
    targets = []
    total_count = 0
    nsites = []
    spacegroups = []
    for i, (graph,nsite,spacegroup,target) in enumerate(dataset_list):
        nodes.append(graph.nodes)
        edge_sources.append(graph.edge_sources + total_count)
        edge_targets.append(graph.edge_targets + total_count)
        graph_indices += [i] * len(graph)
        node_counts.append(len(graph))
        targets.append(target)
        total_count += len(graph)
        nsites.append(nsite)
        spacegroups.append(spacegroup)
    nodes = np.concatenate(nodes)
    edge_sources = np.concatenate(edge_sources)
    edge_targets = np.concatenate(edge_targets)
    input = [nodes, edge_sources, edge_targets, graph_indices, node_counts,nsites,spacegroups]

    targets = torch.Tensor(targets)
    return input, targets

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self):
        #element_number -> element_coordinate -> embedding
        # with open(elem_embedding_file) as f:
        #     elem_embedding = json.load(f)
        coordinate = []
        atom_types = set(postable.keys())
        for i in  atom_types:
            coordinate.append(postable[i])


        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key,[x,y] in postable.items():
            tconst_temp = copy.deepcopy(tconst)
            tconst_temp[x,y] = 1
            self._embedding[key] = np.array(tconst_temp,dtype=float)

def load_target(target_name, file_path):
    df = pd.read_csv(file_path)
    graph_names = df["name"].values
    targets = df[target_name].values
    spacegroup = df['spacegroup'].values
    nsite = df['nsites'].values
    return graph_names, targets, spacegroup, nsite
def makeperiodtable():
    temptable = np.zeros((6, 18), dtype=np.float32)
    temptable[0, 1:17] = -1
    temptable[1, 2:12] = -1
    temptable[2, 3:12] = -1
    temptable[5, 2] = -1
    temptable[5, 15:18] = -1
    nonvalidpos = np.where(temptable == -1)

    elem_pos = {1: (0, 0), 2: (0, 17), 3: (1, 0), 4: (1, 1), 11: (2, 0), 12: (2, 1), 55: (5, 0), 56: (5, 1)}
    telem_pos = {i: (1, i + 7) for i in range(5, 11)}
    elem_pos.update(telem_pos)
    telem_pos = {i: (2, i - 1) for i in range(13, 19)}
    elem_pos.update(telem_pos)
    telem_pos = {i: (3, i - 19) for i in range(19, 37)}
    elem_pos.update(telem_pos)
    telem_pos = {i: (4, i - 37) for i in range(37, 55)}
    elem_pos.update(telem_pos)
    telem_pos = {i: (5, i - 69) for i in range(72, 84)}
    elem_pos.update(telem_pos)

    nodes_info = {}
    for key in elem_pos.keys():
        ty, tx = elem_pos[key]
        ttable = -np.ones((6, 18), dtype=np.float32) / 68
        ttable[ty, tx] = 1
        ttable[nonvalidpos] = 0
        nodes_info[key] = ttable

    return nodes_info

def load_graph_data(file_path):
    try:
        graphs = np.load(file_path, allow_pickle=True)['graph_dict'].item()
    except UnicodeError:
        graphs = np.load(file_path, allow_pickle=True, encoding='latin1')['graph_dict'].item()
        graphs = { k.decode() : v for k, v in graphs.items() }
    return graphs
class Graph(object):
    def __init__(self, graph, node_vectors):
        self.nodes, self.neighbors = graph
        self.neighbors = list(self.neighbors)

        # n_types = len(node_vectors)
        n_nodes = len(self.nodes)

        # Make node representations

        self.nodes = [node_vectors[i] for i in self.nodes]
        self.nodes = np.array(self.nodes, dtype=np.float32)
        #nodes = np.stack(self.nodes, axis=0)
        self.nodes = np.stack(self.nodes, axis=0)
        self.edge_sources = np.concatenate([[i] * len(self.neighbors[i]) for i in range(n_nodes)])
        self.edge_targets = np.concatenate(self.neighbors)

    def __len__(self):
        return len(self.nodes)


class GraphDataset(Dataset):
    def __init__(self, path, target_name):
        super(GraphDataset, self).__init__()
        # import ipdb
        # ipdb.set_trace()

        self.path = path
        target_path = os.path.join(path, "targets.csv")
        self.graph_names, self.targets,self.spacegroup, self.nsites = load_target(target_name, target_path)


        node_vectors = AtomCustomJSONInitializer()
        self.node_vectors = []
        for  i in range(100):
            self.node_vectors.append(node_vectors.get_atom_fea(i+1))


        graph_data_path = os.path.join(path, "graph_data.npz")
        self.graph_data = load_graph_data(graph_data_path)


        tmpd = dict(zip(self.graph_names, self.targets))
        tmpd2 = dict(zip(self.graph_names, self.spacegroup))
        tmpd3 = dict(zip(self.graph_names, self.nsites))
        self.graph_names = self.graph_data.keys()
        self.targets = [tmpd[tname] for tname in self.graph_names]
        self.spacegroup = [tmpd2[tname] for tname in self.graph_names]
        self.nsites = [tmpd3[tname] for tname in self.graph_names]
        self.graph_data = [Graph(self.graph_data[name], self.node_vectors)
                           for name in self.graph_names]

    def __getitem__(self, index):
        return self.graph_data[index],self.nsites[index],self.spacegroup[index], self.targets[index]

    def __len__(self):
        return len(self.graph_names)


def graph_collate(batch):
    nodes = []
    edge_sources = []
    edge_targets = []
    graph_indices = []
    node_counts = []
    targets = []
    total_count = 0
    for i, (graph, target) in enumerate(batch):
        nodes.append(graph.nodes)
        edge_sources.append(graph.edge_sources + total_count)
        edge_targets.append(graph.edge_targets + total_count)
        graph_indices += [i] * len(graph)
        node_counts.append(len(graph))
        targets.append(target)
        total_count += len(graph)

    nodes = np.concatenate(nodes)
    edge_sources = np.concatenate(edge_sources)
    edge_targets = np.concatenate(edge_targets)

    input = GGNNInput(nodes, edge_sources, edge_targets, graph_indices, node_counts)
    targets = torch.Tensor(targets)
    return input, targets