import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
# from model_local import DemosaicCNN
from model import DemosaicNet1D
from data import prepare_data

# Load and prepare the dataset
img_dir = './img'
patches, gt_data = prepare_data(img_dir)

print("finish pre")
# Convert to PyTorch tensors
patches_tensor = torch.tensor(patches, dtype=torch.float)
gt_data_tensor = torch.tensor(gt_data, dtype=torch.float)
print("tensors")
# Create a TensorDataset and DataLoader
dataset = TensorDataset(patches_tensor, gt_data_tensor)
dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)
print("dataset loaded")

# Initialize the model, loss function, and optimizer
model = DemosaicNet1D()
criterion = nn.MSELoss()
optimizer = optim.Adagrad(model.parameters(), lr=0.002)
# optim.Adam(model.parameters(), lr=0.04)
print("model created")
# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'demosaic_cnn.pth')
