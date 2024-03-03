import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm

class GraphNet(object):
    """
    Model Container and Trainer for Graph Neural Network
    """
    def __init__(
        self,
        kernel_model: nn.Module,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.kernel_model = kernel_model
        self.device = device
        self.kernel_model.to(device)
        
    def train(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        val_loader: DataLoader = None,
        epochs: int = 100,
        lr: float = 1e-2,
        lr_scheduler: bool = False,
        log_interval: int = 10,
    ):
        optimizer = optimizer(self.kernel_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
        self.kernel_model.train()
        
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            loss_list = []
            pbar = tqdm(train_loader, desc="Training Loss: ...., Validation Loss: ....")
            for batch_num, batch_data in enumerate(pbar):
                optimizer.zero_grad()
                batch_data = batch_data.to(self.device)
                batch_target = batch_data.y
                batch_output = self.kernel_model(batch_data)
                loss = criterion(batch_output, batch_target)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                
                if (batch_num % log_interval == 0 and batch_num > 0) or (batch_num == len(train_loader) - 1):
                    if batch_num == len(train_loader) - 1:
                        cur_loss = np.mean(loss_list)
                        val_loss = self.evaluate(val_loader, criterion)
                        pbar.set_description(f"Training Loss: {cur_loss:.2f}, Validation Loss: {val_loss:.2f}")
                    else:
                        cur_loss = np.mean(loss_list)
                        pbar.set_description(f"Training Loss: {cur_loss:.2f}, Validation Loss: ....")
            if lr_scheduler:             
                scheduler.step(val_loss)
                
    def evaluate(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
    ):
        self.kernel_model.eval()
        loss_list = []
        with torch.no_grad():
            for batch_num, batch_data in enumerate(data_loader):
                batch_data = batch_data.to(self.device)
                batch_target = batch_data.y
                batch_output = self.kernel_model(batch_data)
                loss = criterion(batch_output, batch_target)
                loss_list.append(loss.item())
        self.kernel_model.train()
        return np.mean(loss_list)