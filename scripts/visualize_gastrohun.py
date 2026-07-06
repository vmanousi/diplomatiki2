import matplotlib.pyplot as plt
from torchvision import transforms

from src.data.gastrohun_dataset import GastroHUNDataset

images_root = "/home/vasia/Downloads/Labeled_Images"
csv_path = "/home/vasia/Downloads/official_splits/image_classification.csv"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

ds = GastroHUNDataset(
    images_root=images_root,
    csv_path=csv_path,
    split="Train",
    label_column="Complete agreement",
    transform=transform,
)

plt.figure(figsize=(10, 8))

for i in range(9):
    image, label = ds[i]
    label_name = ds.idx_to_label[label]

    image = image.permute(1, 2, 0)

    plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(label_name)
    plt.axis("off")

plt.tight_layout()
plt.show()