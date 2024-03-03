import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum, scatter_max
from graphnn.models.gcn.gcn import GateMLP

class GANMainBlock(nn.Module):
    """
    Main Block for Graph Attention Network
    """
    def __init__(
        self,
        hidden_dim: int,
    ):
        super(GANMainBlock, self).__init__()
        self.node_linear = nn.Linear(hidden_dim, hidden_dim)
        self.edge_linear = nn.Linear(3*hidden_dim, 1)
        
    def forward(self, node_attr: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor, num_nodes: torch.Tensor):
        node_attr = self.node_linear(node_attr)
        node_edge_concat = torch.concat((edge_attr, node_attr[edge_index[0]], node_attr[edge_index[1]]), dim=1)
        edge_attr = nn.LeakyReLU(negative_slope=0.2)(self.edge_linear(node_edge_concat)).view(-1)
        edge_scatter_max = scatter_max(edge_attr, edge_index[0], dim=0, dim_size=node_attr.size(0))[0]
        edge_scatter_max = edge_scatter_max[edge_index[0].tolist()]
        exp_edge_attr = torch.exp(edge_attr-edge_scatter_max)
        expsum_edge_attr = scatter_sum(exp_edge_attr, edge_index[0], dim=0, dim_size=node_attr.size(0))
        exp_sum_edge_attr = expsum_edge_attr[edge_index[0].tolist()]
        softmax_a = exp_edge_attr / exp_sum_edge_attr
        neighbor_node_attr = node_attr[edge_index[1].tolist(), :]
        softmax_anode = softmax_a.unsqueeze(1).expand_as(neighbor_node_attr) * neighbor_node_attr
        adjacent_attr_sum = scatter_sum(softmax_anode, edge_index[0], dim=0, dim_size=node_attr.size(0))
        node_attr = nn.LeakyReLU(negative_slope=0.2)(node_attr + adjacent_attr_sum)
        return node_attr
        
        

class GraphAttnNet(nn.Module):
    """
    Graph Attention Network
    paper: https://arxiv.org/pdf/1710.10903.pdf
    """
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_class: int,
        num_layer: int,
    ):
        super(GraphAttnNet, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.num_layer = num_layer
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.mainblocks = nn.ModuleList(
            [
                GANMainBlock(
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
        edge_attr = batch.edge_attr
        edge_index = batch.edge_index
        num_atoms = batch.num_atoms
        node_attr = self.node_encoder(node_attr)
        edge_attr = self.edge_encoder(edge_attr)
        for mainblock in self.mainblocks:
            node_attr = mainblock(
                node_attr = node_attr,
                edge_attr = edge_attr,
                edge_index = edge_index,
                num_nodes = num_atoms,
            )
            
        node_attr = self.conv_layer(node_attr)
        graph_mask = torch.arange(num_atoms.shape[0]).to(node_attr.device)
        graph_mask = graph_mask.repeat_interleave(num_atoms)
        graph_attr = scatter_mean(node_attr, graph_mask, dim=0)
        output = self.final_layer(graph_attr)
        return output