import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_sum

class ReLuMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super(ReLuMLP, self).__init__()
        self.module_list = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.LeakyReLU()
            ]
        )
    
    def forward(self, x: torch.Tensor):
        for layer in self.module_list:
            x = layer(x)
        return x
    
class SigmoidMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super(SigmoidMLP, self).__init__()
        self.module_list = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid()
            ]
        )
    
    def forward(self, x: torch.Tensor):
        for layer in self.module_list:
            x = layer(x)
        return x

class GateMLP(nn.Module):
    """
    GateMLP(x) = ReLuMLP(x) * SigmoidMLP(x)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super(GateMLP, self).__init__()
        self.relumlp = ReLuMLP(input_dim, hidden_dim, output_dim)
        self.sigmoidmlp = SigmoidMLP(input_dim, hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor):
        relu_x = self.relumlp(x)
        sigmoid_x = self.sigmoidmlp(x)
        output = relu_x * sigmoid_x
        return output
        

class GCNMainBlock(nn.Module):
    """
    A First Version of Main Block for GraphConvNet. Follow the Paper: paper: https://arxiv.org/pdf/1609.02907.pdf
    Main Block is One of the Most Important Module for GCN
    """
    def __init__(
        self,
        hidden_dim: int,
    ):
        super(GCNMainBlock, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, node_attr: torch.Tensor, edge_index: torch.Tensor):
        node_attr = self.linear(node_attr)
        ## Scatter node_attr according to the edge_index
        adjacent_node_attr = scatter_sum(node_attr[edge_index[1]], edge_index[0], dim=0, dim_size=node_attr.size(0))
        node_attr = node_attr + adjacent_node_attr
        degree = scatter_sum(torch.ones_like(edge_index[0]), edge_index[0], dim=0, dim_size=node_attr.size(0))
        degree = (degree + 1).unsqueeze(1).expand_as(node_attr)
        node_attr = node_attr / degree
        return node_attr
        

class GraphConvNet(nn.Module):
    """
    An Implementation of Graph Convolutional Network (GCN) for Graph Classification
    Using OriGCNMainBlock
    Paper: https://arxiv.org/pdf/1609.02907.pdf
    """
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_class: int,
        num_layer: int,
    ):
        super(GraphConvNet, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.num_layer = num_layer
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.mainblocks = nn.ModuleList(
            [
                GCNMainBlock(
                    hidden_dim = hidden_dim,
                )
                for _ in range(num_layer)
            ]
        )
        self.conv_layer = GateMLP(
            input_dim = hidden_dim,
            hidden_dim = hidden_dim,
            output_dim = hidden_dim,
        )
        self.final_layer = nn.Linear(hidden_dim, num_class)
    
    def forward(self, batch):
        node_attr = batch.node_attr
        edge_index = batch.edge_index
        num_atoms = batch.num_atoms
        node_attr = self.node_encoder(node_attr)
        for mainblock in self.mainblocks:
            node_attr = mainblock(node_attr, edge_index)
            
        node_attr = self.conv_layer(node_attr)
        graph_mask = torch.arange(num_atoms.shape[0]).to(node_attr.device)
        graph_mask = graph_mask.repeat_interleave(num_atoms)
        graph_attr = scatter_mean(node_attr, graph_mask, dim=0)
        output = self.final_layer(graph_attr)
        return output