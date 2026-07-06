import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models

from src.data.gastrohun_dataset import GastroHUNDataset
from src.evaluation.evaluate import evaluate_model


images_root = "/home/vasia/Downloads/Labeled_Images"
csv_path = "/home/vasia/Downloads/official_splits/image_classification.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_ds = GastroHUNDataset(
    images_root,
    csv_path,
    split="Test",
    label_column="Complete agreement",
    transform=transform,
)

test_loader = DataLoader(
    test_ds,
    batch_size=16,
    shuffle=False,
    num_workers=2,
)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 23)

checkpoint = torch.load(
    "experiments/exp01_resnet18/checkpoints/best_model.pt",
    map_location=device,
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)

evaluate_model(
    model=model,
    dataloader=test_loader,
    device=device,
    idx_to_label=test_ds.idx_to_label,
    output_dir="experiments/exp01_resnet18/evaluation",
)