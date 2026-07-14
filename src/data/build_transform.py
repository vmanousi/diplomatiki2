from torchvision import transforms

from src.data.dino_transform import DINOMultiCropTransform


def build_transform(config):
    """
    Build the image transformation pipeline from the full YAML config.
    """

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

    if transform_name == "dino_multicrop":
        return DINOMultiCropTransform(
            global_crop_size=transform_cfg.get(
                "global_crop_size",
                224,
            ),
            local_crop_size=transform_cfg.get(
                "local_crop_size",
                96,
            ),
            global_crop_scale=tuple(
                transform_cfg.get(
                    "global_crop_scale",
                    [0.4, 1.0],
                )
            ),
            local_crop_scale=tuple(
                transform_cfg.get(
                    "local_crop_scale",
                    [0.05, 0.4],
                )
            ),
            num_local_crops=transform_cfg.get(
                "num_local_crops",
                6,
            ),
        )

    raise ValueError(f"Unknown transform: {transform_name}")
