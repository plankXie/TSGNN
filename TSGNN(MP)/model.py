from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
class FastEdgeNetwork(nn.Module):
    """
    Fast Edge Network layer
    """
    def __init__(self, in_features, out_features, n_layers, activation=nn.ELU(),
                 net_type=0, use_batch_norm=False, bias=False,
                 use_shortcut=False):
        super(FastEdgeNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        filtersz=3
        self.node_linears = [nn.Conv2d(in_features, out_features, filtersz,padding=1, bias=not use_batch_norm and bias)]
        self.edge_linears = [nn.Conv2d(in_features, out_features, filtersz,padding=1,bias=not use_batch_norm and bias)]
        self.edge_linears += [nn.Conv2d(out_features, out_features,filtersz,padding=1,bias=not use_batch_norm and bias)
                                for _ in range(n_layers-1)]
        self.node_linears = nn.ModuleList(self.node_linears)
        self.edge_linears = nn.ModuleList(self.edge_linears)
        self.activation = activation
        self.net_type = net_type
    def forward(self, input, edge_sources, e):
        z = e
        for node_linear, edge_linear in zip(self.node_linears,
            self.edge_linears):
            z = edge_linear(z)
            if self.net_type == 0:
                th = node_linear(input.clone())[edge_sources]
                z = self.activation(th * z)
            else:
                th = self.activation(node_linear(input.clone()))[edge_sources]
                z = th * self.activation(z)
        return z

class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()


        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.linear = nn.Conv2d(16,32,kernel_size=3,padding=1)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm2d(16)
        self.softplus2 = nn.Softplus()
        self.edgenet = FastEdgeNetwork(16,16,3)
    def forward(self, atom_in_fea, edge_sources, edge_targets):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        # N, M = nbr_fea_idx.shape
        # # convolution
        # atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        # total_nbr_fea = torch.cat(
        #     [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
        #      atom_nbr_fea, nbr_fea], dim=2)
        # total_gated_fea = self.fc_full(total_nbr_fea)
        # total_gated_fea = self.bn1(total_gated_fea.view(
        #     -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        th = atom_in_fea[edge_targets]
        th = self.edgenet(atom_in_fea,edge_sources,th)
        th = self.linear(th)
        nbr_filter, nbr_core = th.chunk(2, dim=1)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        output = atom_in_fea.clone()
        output.index_add_(0, edge_sources, nbr_core * nbr_filter)
        nbr_sumed = self.bn2(output)
        out = self.softplus2(atom_in_fea + nbr_sumed)

        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        # self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.embedding = nn.Conv2d(1,16,3,padding=1)
        self.total_embedding = nn.Conv2d(16, 16, 3, padding=1)
        self.activation = nn.Softplus()
        self.space_feature = space_feature()
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Conv2d(16, 16,3,padding=1)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        self.fc1 = FullConnection(16*7*18,256)
        self.fc2 = FullConnection(256,32)
        self.regression = LinearRegression(32,1)
        self.space_encode = Space_encode()

        self.flatten = nn.Flatten()
        self.bn = nn.BatchNorm2d(16)
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,cry_coord):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        space_encode = self.space_encode(cry_coord)
        atom_fea = self.embedding(atom_fea)
        atom_fea = self.activation(atom_fea)
        # atom_fea = self.total_embedding(atom_fea)
        # atom_fea = self.bn(atom_fea)
        # atom_fea = self.activation(atom_fea)
        # atom_fea = self.total_embedding(atom_fea)
        # atom_fea = self.activation(atom_fea)
        space_encode = self.space_feature(space_encode)

        edge_sources = torch.tensor(np.concatenate([[i] * 12 for i in range(atom_fea.size(0))]))
        edge_sources = edge_sources.to('cuda')
        edge_targets = torch.tensor(nbr_fea_idx.view(-1))
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, edge_sources,edge_targets)
        graph_indices = torch.tensor(np.concatenate([[i] * len(crystal_atom_idx[i]) for i in range(len(crystal_atom_idx))]))
        graph_indices = graph_indices.to('cuda')
        nodes_counts = torch.tensor(np.concatenate([[len(crystal_atom_idx[i])] for i in range(len(crystal_atom_idx))]))
        nodes_counts = nodes_counts.to('cuda')
        crys_fea = self.pooling(atom_fea, graph_indices,nodes_counts)
        space_encode = self.pooling(space_encode,graph_indices,nodes_counts)
        space_encode = self.flatten(space_encode)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
          for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        crys_fea = self.fc1(crys_fea)
        crys_fea = self.fc2(crys_fea)
        out = self.regression(crys_fea,space_encode)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, input, graph_indices, node_counts):
            graph_count = node_counts.size(0)
            fc, fh, fw = input.size()[1:]
            g = torch.zeros(graph_count, fc, fh, fw).to(input.device)
            g.index_add_(0, graph_indices, input)

            tn = torch.unsqueeze(node_counts, 1)
            tn = torch.unsqueeze(tn, 2)
            tn = torch.unsqueeze(tn, 3)
            output = g / tn
            return output

    def pooling2(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)

class LinearRegression(nn.Module):
    """
    Linear Regression layer
    """
    def __init__(self, in_features, out_features=1):
        super(LinearRegression, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(1024,512)
        self.linear4 = nn.Linear(512,128)
        self.linear3 = nn.Linear(128, 1)
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.eps2 = nn.Parameter(torch.Tensor([0]))
    def forward(self, input,space_encode):
        output = self.linear(input)
        space_encode = self.linear2(space_encode)
        space_encode = self.linear4(space_encode)
        output = self.linear3((1+self.eps)*output+self.eps2*space_encode)
        # output = self.linear3(space_encode)
        return output

class space_feature(nn.Module):
    """
    Linear Regression layer
    """
    def __init__(self):
        super(space_feature, self).__init__()
        self.linear1 = nn.Conv2d(16,32,5,padding=0)
        self.linear2 = nn.Conv2d(32, 64,5,padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.max3 = nn.MaxPool2d(5)
        self.activation = nn.Softplus()
    def forward(self,input):
        output = self.linear1(input)
        output = self.bn1(output)
        output = self.activation(output)
        output = self.linear2(output)
        output = self.bn2(output)
        output = self.max3(output)
        output = self.activation(output)
        return output
class Space_encode(nn.Module):
    """
    Linear Regression layer
    """
    def __init__(self):
        super(Space_encode, self).__init__()

        self.linear1 = nn.Linear(12,32)
        self.linear2 = nn.Linear(12, 32)
        self.linear3 = nn.Conv2d(1, 16,3,padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.activation = nn.Softplus()
    def forward(self, input):
        row = self.linear1(input)
        row_array = row.to('cpu')
        row_array = row_array.detach().numpy()
        column = self.linear2(input)
        column_array = column.to('cpu')
        column_array = column_array.detach().numpy()
        row = row[:,:,np.newaxis]
        column = column[:,np.newaxis,:]
        output = torch.bmm(row,column)
        output_array = output.to('cpu')
        output_array = output_array.detach().numpy()
        output_array = output_array[0]
        data_df = pd.DataFrame(output_array)
        data_df.to_excel('numpy_data.xlsx', index=False,float_format='%.5f')
        output = output[:, np.newaxis, :, :]# 关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入 # 关键4        output = output[:, np.newaxis, :, :]
        output = self.linear3(output)
        output = self.bn(output)
        output = self.activation(output)
        return output

class FullConnection(nn.Module):
    """
    Full Connection layer
    """
    def __init__(self, in_features, out_features, activation=nn.Softplus(),
                 use_batch_norm=False, bias=True):
        super(FullConnection, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if use_batch_norm:
            self.linear = nn.Linear(in_features, out_features, bias=False) #?
            self.activation = nn.Sequential(nn.BatchNorm1d(out_features), activation)
        else:
            self.linear = nn.Linear(in_features, out_features,bias=bias)
            self.activation = activation

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, input):
        tinput = input.view(-1, self.num_flat_features(input))
        output = self.activation(self.linear(tinput))
        return output

