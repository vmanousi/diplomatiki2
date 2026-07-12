from src.data.gastrovision_webdataset import build_gastrovision_webdataset

root = "data/Gastrovision_webdataset"

ds = build_gastrovision_webdataset(root=root, split="train")

for image, label in ds:
    print("Image shape:", image.shape)
    print("Label:", label)
    break
