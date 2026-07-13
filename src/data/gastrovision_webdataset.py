from pathlib import Path
import glob

import webdataset as wds
from torchvision import transforms


def build_gastrovision_webdataset(root, split, image_size=224):
    root = Path(root)

    shards = sorted(glob.glob(str(root / split / "*.tar")))

    if len(shards) == 0:
        raise FileNotFoundError(f"No tar shards found in: {root / split}")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    dataset = (
        wds.WebDataset(shards, shardshuffle=False)
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(transform, lambda x: int(x))
    )

    return dataset
