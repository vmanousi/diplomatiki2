from torchvision import transforms


def build_transform(config):
    transform_cfg = config.get("transforms", {})
    transform_name = transform_cfg.get("name", "supervised_basic")

    image_size = transform_cfg.get(
        "image_size",
        config.get("dataset", {}).get("image_size", 224),
    )

    if transform_name == "supervised_basic":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    raise ValueError(f"Unknown transform: {transform_name}")
