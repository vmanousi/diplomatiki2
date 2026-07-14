from pathlib import Path
import glob

import webdataset as wds
from torchvision import transforms


def build_gastrovision_webdataset(
    root,
    split,
    image_size=224,
    transform=None,
):
    """
    Build the GastroVision WebDataset.

    Parameters
    ----------
    root:
        Root directory containing train/, val/ or test/ TAR shards.
    split:
        Dataset split, for example "train" or "val".
    image_size:
        Used only when no external transform is provided.
    transform:
        Optional image transform. This may be a normal supervised transform
        or a DINO multi-crop transform.
    """
    root = Path(root)

    shards = sorted(
        glob.glob(str(root / split / "*.tar"))
    )

    if len(shards) == 0:
        raise FileNotFoundError(
            f"No TAR shards found in: {root / split}"
        )

    # Preserve the previous supervised behaviour when no transform is passed.
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    dataset = (
        wds.WebDataset(
            shards,
            shardshuffle=False,
        )
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(
            transform,
            lambda label: int(label),
        )
    )

    return dataset
