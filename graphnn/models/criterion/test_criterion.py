import torch
import torch.nn as nn


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()
        
    def forward(self, output, target):
        output_pred = torch.argmax(output, dim=-1)
        acc_num = torch.sum(output_pred == target)
        acc_ratio = acc_num / output.shape[0]
        return acc_ratio