from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
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

        th = atom_in_fea[edge_targets]
        th = self.edgenet(atom_in_fea,edge_sources,th)
        th = self.linear(th)
        nbr_filter, nbr_core = th.chunk(2, dim=1)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        output = atom_in_fea.clone()
        output.index_add_(0, edge_sources, nbr_core * nbr_filter)
        nbr_sumed = self.bn2(output)
        out = atom_in_fea + nbr_sumed

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
        self.fc3 = nn.Linear(144,1)
        self.activation = nn.Softplus()
        self.space_encode = Space_encode()
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
        self.fc1 = FullConnection(16*7*18,32)
        self.fc2 = FullConnection(32,32)
        self.regression = LinearRegression(32,1)
        self.flatten = nn.Flatten()
        self.regression2 = nn.Linear(1,1)
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.eps2 = nn.Parameter(torch.Tensor([0]))
    def forward(self, nodes, edge_sources, edge_targets, graph_indices, node_counts,nsites,spacegroups):
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
        nodes = nodes[:, np.newaxis, :, :]
        atom_fea = self.embedding(nodes)
        atom_fea = self.activation(atom_fea)
        nsites = torch.unsqueeze(nsites,dim=1)
        spacegroups = torch.unsqueeze(spacegroups,dim=1)
        space_origin = torch.cat((nsites,spacegroups),dim=1)
        space_origin = space_origin.to(torch.float32)
        space_embed = self.space_encode(space_origin)
        space_feature = self.space_feature(space_embed)
        space_feature = self.flatten(space_feature)
        space_feature = self.activation(self.fc3(space_feature))

        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, edge_sources,edge_targets)

        crys_fea = self.pooling(atom_fea, graph_indices,node_counts)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
          for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        crys_fea = self.fc1(crys_fea)
        crys_fea = self.fc2(crys_fea)
        out = self.regression(crys_fea)
        out = self.regression2((1+self.eps)*out+self.eps2*space_feature)
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


class space_feature(nn.Module):
    """
    Linear Regression layer
    """
    def __init__(self):
        super(space_feature, self).__init__()
        self.linear1 = nn.Conv2d(16,16,3,padding=1)
        self.linear2 = nn.Conv2d(16, 16,3,padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.max3 = nn.MaxPool2d(5)
        self.activation = nn.Softplus()
    def forward(self,input):
        output = self.linear1(input)
        output = self.bn1(output)
        output = self.activation(output)
        output = self.linear2(input)
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

        self.linear1 = nn.Linear(2,16)
        self.linear2 = nn.Linear(2, 16)
        self.linear3 = nn.Conv2d(1, 16,3,padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.activation = nn.Softplus()
    def forward(self, input):
        row = self.linear1(input)
        column = self.linear2(input)
        row = row[:,:,np.newaxis]
        column = column[:,np.newaxis,:]
        output = torch.bmm(row,column)
        output = output[:, np.newaxis, :, :]
        output = self.linear3(output)
        output = self.bn(output)
        output = self.activation(output)
        return output
class LinearRegression(nn.Module):
    """
    Linear Regression layer
    """
    def __init__(self, in_features, out_features=1):
        super(LinearRegression, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input):
        output = self.linear(input)
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


