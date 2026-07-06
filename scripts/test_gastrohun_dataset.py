from torchvision import transforms
from src.data.gastrohun_dataset import GastroHUNDataset

images_root = "/home/vasia/Downloads/Labeled_Images"
csv_path = "/home/vasia/Downloads/official_splits/image_classification.csv"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

for split in ["Train", "Validation", "Test"]:
    ds = GastroHUNDataset(
        images_root=images_root,
        csv_path=csv_path,
        split=split,
        label_column="Complete agreement",
        transform=transform,
    )

    print(split)
    print("Images:", len(ds))
    print("Classes:", ds.label_to_idx)
    image, label = ds[0]
    print("Image shape:", image.shape)
    print("First label:", label)
    print()