from src.data.gastrohun_dataset import GastroHUNDataset
from src.data.gastrovision_webdataset import build_gastrovision_webdataset


def build_dataset(config, split, transform=None):
    dataset_cfg = config["dataset"]
    dataset_name = dataset_cfg.get("name", "gastrohun")

    if dataset_name == "gastrohun":
        split_map = {
            "train": "Train",
            "val": "Validation",
            "test": "Test",
        }

        return GastroHUNDataset(
            images_root=dataset_cfg["images_root"],
            csv_path=dataset_cfg["csv_path"],
            split=split_map[split],
            label_column=dataset_cfg["label_column"],
            transform=transform,
        )

    elif dataset_name == "gastrovision_webdataset":
        return build_gastrovision_webdataset(
            root=dataset_cfg["root"],
            split=split,
            image_size=dataset_cfg.get("image_size", 224),
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
