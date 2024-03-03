import torch
from datasets import load_dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np

def get_dataloader(
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    assert train_ratio + val_ratio + test_ratio == 1, "The sum of train_ratio, val_ratio, and test_ratio must be 1."
    dataset_hf = load_dataset("graphs-datasets/AIDS")
    graph_list = []
    for data in dataset_hf["full"]:
        node_attr = torch.tensor(data["node_feat"], dtype=torch.float32)
        edge_attr = torch.tensor(data["edge_attr"], dtype=torch.float32)
        num_atoms = torch.tensor(node_attr.shape[0], dtype=torch.long)
        edge_index = torch.tensor(data["edge_index"])
        y = torch.tensor(data["y"], dtype=torch.long)
        graph_list.append(
            Data(
                node_attr=node_attr,
                edge_attr=edge_attr,
                edge_index=edge_index,
                num_atoms=num_atoms,
                y=y,
            )
        )
    indices = np.random.choice(len(graph_list), len(graph_list), replace=False)
    graph_list = [graph_list[i] for i in indices]
    train_graphs = graph_list[: int(len(graph_list) * train_ratio)]
    val_graphs = graph_list[int(len(graph_list) * train_ratio) : int(len(graph_list) * (train_ratio + val_ratio))]
    test_graphs = graph_list[int(len(graph_list) * (train_ratio + val_ratio)) :]
    print(f"Len of Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloader(batch_size=2)
    for batch in train_loader:
        node_attr = batch.node_attr
        edge_attr = batch.edge_attr
        edge_index = batch.edge_index
        num_node = batch.num_node
        y = batch.y
        print(node_attr.shape)
        print(edge_attr.shape)
        print(edge_index.shape)
        print(num_node.shape)
        print(y.shape)
        # print(edge_index)
        print(num_node)
        print(y)
        break