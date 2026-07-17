from torchvision import transforms

from src.data.dino_transform import DINOMultiCropTransform


def build_transform(config, split="train"):
    """
    Build the image transformation pipeline from the full YAML config.

    split:
        "train", "val", or "test". Augmentation (when the transform name
        requests it) is only ever applied for split="train" — val/test
        always get the same deterministic resize + ToTensor pipeline so
        evaluation numbers stay comparable across experiments.
    """

    transform_cfg = config.get("transforms", {})
    transform_name = transform_cfg.get("name", "supervised_basic")

    image_size = transform_cfg.get(
        "image_size",
        config.get("dataset", {}).get("image_size", 224),
    )

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    if transform_name == "supervised_basic":
        return eval_transform

    if transform_name == "supervised_augmented":
        if split != "train":
            return eval_transform

        return transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=tuple(
                    transform_cfg.get(
                        "random_resized_crop_scale",
                        [0.7, 1.0],
                    )
                ),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(
                transform_cfg.get("random_rotation_degrees", 15)
            ),
            transforms.ColorJitter(
                brightness=transform_cfg.get("color_jitter_brightness", 0.1),
                contrast=transform_cfg.get("color_jitter_contrast", 0.1),
                saturation=transform_cfg.get("color_jitter_saturation", 0.1),
            ),
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
