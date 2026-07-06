import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models

from src.data.gastrohun_dataset import GastroHUNDataset

images_root = "/home/vasia/Downloads/Labeled_Images"
csv_path = "/home/vasia/Downloads/official_splits/image_classification.csv"

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_ds = GastroHUNDataset(images_root, csv_path, "Train", "Complete agreement", transform)
val_ds = GastroHUNDataset(images_root, csv_path, "Validation", "Complete agreement", transform)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 23)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("Starting training on CPU...")

model.train()
for batch_idx, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if batch_idx % 10 == 0:
        print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

    if batch_idx == 20:
        break

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Validation accuracy: {accuracy:.4f}")
print("Done.")