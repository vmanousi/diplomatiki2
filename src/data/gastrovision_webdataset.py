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

    # Only shuffle the training split. Without this, every epoch iterates
    # the shards/samples in the exact same fixed order every time, which
    # hurts SGD/SSL training quality — val/test should stay deterministic.
    # webdataset wants an explicit buffer size here, not a bare bool.
    shuffle_samples = split == "train"

    dataset = wds.WebDataset(
        shards,
        shardshuffle=100 if shuffle_samples else False,
    )

    if shuffle_samples:
        # Shuffles a buffer of raw (still-encoded) samples before decode,
        # so the memory cost stays low while still mixing across shards.
        dataset = dataset.shuffle(1000)

    dataset = (
        dataset
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(
            transform,
            lambda label: int(label),
        )
    )

    return dataset
