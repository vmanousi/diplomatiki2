from torchvision import transforms

from src.data.dino_transform import DINOMultiCropTransform
from src.models.build_model import resolve_backbone_normalization


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
    model_cfg = config.get("model", {})

    image_size = transform_cfg.get(
        "image_size",
        config.get("dataset", {}).get("image_size", 224),
    )

    # A pretrained backbone (ImageNet or a DINO backbone_checkpoint) needs
    # its input normalized to match what it was actually pretrained on.
    # From-scratch models (no pretrained weights involved) have no such
    # requirement, so this is None and Normalize is simply omitted.
    normalization = None

    if transform_name in {"supervised_basic", "supervised_augmented"}:
        normalization = resolve_backbone_normalization(
            model_name=model_cfg.get("name"),
            pretrained=model_cfg.get("pretrained", False),
            backbone_checkpoint=model_cfg.get("backbone_checkpoint"),
        )

    eval_steps = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]

    if normalization is not None:
        mean, std = normalization
        eval_steps.append(transforms.Normalize(mean=mean, std=std))

    eval_transform = transforms.Compose(eval_steps)

    if transform_name == "supervised_basic":
        return eval_transform

    if transform_name == "supervised_augmented":
        if split != "train":
            return eval_transform

        train_steps = [
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
        ]

        if normalization is not None:
            mean, std = normalization
            train_steps.append(transforms.Normalize(mean=mean, std=std))

        return transforms.Compose(train_steps)

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
