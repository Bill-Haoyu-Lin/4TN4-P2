import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
# from model_local import DemosaicCNN
from model import DemosaicNet1D
from data import prepare_data
from torch.cuda.amp import GradScaler, autocast

# Load and prepare the dataset
img_dir = './img'
patches, gt_data = prepare_data(img_dir)
device = torch.device('cuda:0')

print("finish pre")
# Convert to PyTorch tensors
patches_tensor = torch.tensor(patches, dtype=torch.float).to(device)
gt_data_tensor = torch.tensor(gt_data, dtype=torch.float).to(device)
print("tensors")
# Create a TensorDataset and DataLoader
dataset = TensorDataset(patches_tensor, gt_data_tensor)
dataloader = DataLoader(dataset, batch_size=16384, shuffle=True)
print("dataset loaded")

# Initialize the model, loss function, and optimizer
model = DemosaicNet1D().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adagrad(model.parameters(), lr=0.0005)
# optim.Adam(model.parameters(), lr=0.04)
print("model created")
# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    scaler = GradScaler()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'demosaic_cnn.pth')
