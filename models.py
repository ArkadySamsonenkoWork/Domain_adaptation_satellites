from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

class ClassificatorModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.class_layer = nn.Sequential(nn.MaxPool2d(kernel_size=8),
                                         nn.Flatten(),
                                         nn.Linear(2048, 2),
                                        )

    def forward(self, input):
        input = self.encoder(input)[-1]
        return self.class_layer(input)

    def train(self, flag=True):
        super().train(flag)
        self.encoder.eval()



class EncoderModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.max_pool = nn.Sequential(nn.MaxPool2d(kernel_size=8),
                                         nn.Flatten(),
                                        )   

    @torch.no_grad()
    def forward(self, input):
        input = self.encoder(input)[-1]
        out = self.max_pool(input)
        return out

    def train(self, flag=True):
        super().train(flag)
        self.encoder.eval()
            