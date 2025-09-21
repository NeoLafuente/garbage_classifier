import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Subset
import random
from tqdm import tqdm

# Data transforms
transform = models.VGG11_Weights.IMAGENET1K_V1.transforms()

# Load full dataset
full_dataset = datasets.ImageFolder('../../data/raw/Garbage_Dataset_Classification/images/', transform=transform)

# Sample subset of data
# Assuming cats=0, dogs=1 in alphabetical order

cardboard_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 0]
glass_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 1]
metal_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 2]
paper_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 3]
plastic_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 4]
trash_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 5]

# Sample 600 cats and 600 dogs (500 train + 100 test each)
random.seed(42)  # For reproducibility

cardboard_sample = random.sample(cardboard_indices, 600)
glass_sample = random.sample(glass_indices, 600)
metal_sample = random.sample(metal_indices, 600)
paper_sample = random.sample(paper_indices, 600)
plastic_sample = random.sample(plastic_indices, 600)
trash_sample = random.sample(trash_indices, 600)

# Split into train/test
train_indices = cardboard_sample[:500] + glass_sample[:500] + metal_sample[:500] + paper_sample[:500] + plastic_sample[:500] + trash_sample[:500]
test_indices = cardboard_sample[500:] + glass_sample[500:] + metal_sample[500:] + paper_sample[500:] + plastic_sample[:500] + trash_sample[500:]

# Create subset datasets
train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)

# Data loaders, one for train and one for test. Obviously, the test has different parameters
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)# I'm a cat sleeping on a keyboard

# Model - using VGG11. This network is pretrained with the weights from
# https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg11.html
model = models.vgg11(weights = models.VGG11_Weights.IMAGENET1K_V1)# Meowwwwwwwww prrrrrr
model.classifier[6] = nn.Linear(4096, 6)  # Change final layer for 2 classes

# Freeze all feature layers, we will only train the final classifier
for param in model.features.parameters():
    param.requires_grad = False

print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# We use a regular ADAM optimizer to adjust the model parameters
opt = optim.Adam(model.parameters()) # ZzzzzzzzzZZZzzzzz Prr prrr Meow
loss_fn = nn.CrossEntropyLoss()

# Training
model.train()
for epoch in range(3):
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/3')
    for xb, yb in progress_bar:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for xb, yb in tqdm(test_loader, desc='Testing'):
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        preds = out.argmax(1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

accuracy = correct / total
print(f"Test accuracy: {accuracy:.4f}")
print(f"Total training samples: {len(train_dataset)}")
print(f"Total test samples: {len(test_dataset)}")

torch.save(model.state_dict(), "../../models/model_vgg11_garbage.pth")
