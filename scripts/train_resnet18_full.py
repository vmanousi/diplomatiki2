import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models

from src.data.gastrohun_dataset import GastroHUNDataset
from src.training.trainer import Trainer


images_root = "/home/vasia/Downloads/Labeled_Images"
csv_path = "/home/vasia/Downloads/official_splits/image_classification.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_ds = GastroHUNDataset(images_root, csv_path, "Train", "Complete agreement", transform)
val_ds = GastroHUNDataset(images_root, csv_path, "Validation", "Complete agreement", transform)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 23)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    checkpoint_dir="experiments/exp01_resnet18/checkpoints",
)

history = trainer.fit(epochs=3)

pd.DataFrame(history).to_csv(
    "experiments/exp01_resnet18/history.csv",
    index=False,
)

print("Training finished.")