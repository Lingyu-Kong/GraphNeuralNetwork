import os
import torch
import torch.nn as nn
from graphnn.data_loader.aids_loader import get_dataloader
from graphnn.models.graphnet import GraphNet
from graphnn.models.gcn.gcn import GraphConvNet
from graphnn.models.gat.gat import GraphAttnNet
from graphnn.models.criterion.test_criterion import Accuracy
import argparse


def main(args):
    train_loader, val_loader, test_loader = get_dataloader(batch_size=args.batch_size)
    
    if args.model_type == "gcn":
        kernel_model = GraphConvNet(
            node_dim = args.node_dim,
            edge_dim = args.edge_dim,
            hidden_dim = args.hidden_dim,
            num_class = args.num_class,
            num_layer = args.num_layer,
        )
    elif args.model_type == "gat":
        kernel_model = GraphAttnNet(
            node_dim = args.node_dim,
            edge_dim = args.edge_dim,
            hidden_dim = args.hidden_dim,
            num_class = args.num_class,
            num_layer = args.num_layer,
        )
    else:
        raise ValueError("Model type not supported")
        
    graphnet = GraphNet(
        kernel_model = kernel_model,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    graphnet.train(
        train_loader = train_loader,
        optimizer = torch.optim.Adam,
        criterion = nn.CrossEntropyLoss(),
        val_loader = val_loader,
        epochs = args.epochs,
        lr = args.lr,
        lr_scheduler = args.lr_scheduler,
        log_interval = args.log_interval,
    )
    
    eval_loss = graphnet.evaluate(
        data_loader = test_loader,
        criterion = nn.CrossEntropyLoss(),
    )
    print("Test Loss:", eval_loss)
    accuracy = graphnet.evaluate(
        data_loader = test_loader,
        criterion = Accuracy(),
    )
    print("Accuracy: ", accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphNet")
    parser.add_argument("--model_type", type=str, default="gat", help="Model type")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--lr_scheduler", type=bool, default=True, help="Use learning rate scheduler")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--node_dim", type=int, default=38, help="Node dimension")
    parser.add_argument("--edge_dim", type=int, default=3, help="Edge dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num_class", type=int, default=2, help="Number of classes")
    parser.add_argument("--num_layer", type=int, default=4, help="Number of message passing layers")
    args = parser.parse_args()
    
    main(args)