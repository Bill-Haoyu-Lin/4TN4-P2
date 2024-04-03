import torch
import torch.nn as nn
import torch.nn.functional as F

class DemosaicCNN(nn.Module):
    def __init__(self):
        super( DemosaicCNN, self).__init__()
        # Reduce the number of channels in each layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Reduced from 64 to 32
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)  # Reduced from 128 to 16
        # Final layer directly outputs 2 values for the 2 missing channels
        self.fc1 = nn.Linear(16 * 5 * 5, 2)  # Adjusted based on the conv2 output 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.fc1(x)
        return x
