import torch
import torch.nn as nn

class DemosaicNet1D(nn.Module):
    def __init__(self):
        super(DemosaicNet1D, self).__init__()
        self.fc1 = nn.Linear(25, 64)  # Input is now a flattened array of 25 elements
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 3)    # Predicting 3 channels for the center pixel

    def forward(self, x):
        x = torch.flatten(x, start_dim=1) # Ensure x is flattened (if not already)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x