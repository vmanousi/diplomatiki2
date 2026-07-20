import gc

import torch

from src.models.build_model import (
    build_model,
    resolve_backbone_normalization,
)


NUM_CLASSES = 23
IMAGE_SIZE = 224
BATCH_SIZE = 2


EXPERIMENTS = [
    {
        "name": "ImageNet ViT-S frozen",
        "model_name": "vit_small",
        "pretrained": True,
        "freeze_backbone": True,
    },
    {
        "name": "ImageNet ViT-S full fine-tuning",
        "model_name": "vit_small",
        "pretrained": True,
        "freeze_backbone": False,
    },
    {
        "name": "DINOv2 ViT-S frozen",
        "model_name": "vit_small_patch14_dinov2.lvd142m",
        "pretrained": True,
        "freeze_backbone": True,
    },
    {
        "name": "DINOv2 ViT-S full fine-tuning",
        "model_name": "vit_small_patch14_dinov2.lvd142m",
        "pretrained": True,
        "freeze_backbone": False,
    },
]


def count_parameters(model):
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total_parameters, trainable_parameters


def validate_model(experiment, device):
    print("\n" + "=" * 72)
    print(experiment["name"])
    print("=" * 72)

    model = build_model(
        model_name=experiment["model_name"],
        num_classes=NUM_CLASSES,
        pretrained=experiment["pretrained"],
        freeze_backbone=experiment["freeze_backbone"],
    )

    model = model.to(device)
    model.eval()

    normalization = resolve_backbone_normalization(
        model_name=experiment["model_name"],
        pretrained=experiment["pretrained"],
        backbone_checkpoint=None,
    )

    total_parameters, trainable_parameters = count_parameters(model)

    dummy_images = torch.randn(
        BATCH_SIZE,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    with torch.inference_mode():
        logits = model(dummy_images)

    expected_shape = (
        BATCH_SIZE,
        NUM_CLASSES,
    )

    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            f"Incorrect output shape: {tuple(logits.shape)}. "
            f"Expected: {expected_shape}."
        )

    if not torch.isfinite(logits).all():
        raise RuntimeError(
            "The model produced non-finite output values."
        )

    if experiment["freeze_backbone"]:
        expected_head_parameters = (
            model.head.in_features * NUM_CLASSES
            + NUM_CLASSES
        )

        if trainable_parameters != expected_head_parameters:
            raise RuntimeError(
                "Frozen model has an unexpected number of "
                f"trainable parameters: {trainable_parameters:,}. "
                f"Expected: {expected_head_parameters:,}."
            )
    else:
        if trainable_parameters != total_parameters:
            raise RuntimeError(
                "Full fine-tuning was requested, but some "
                "parameters are frozen."
            )

    print(f"Model identifier:       {experiment['model_name']}")
    print(f"Device:                 {device}")
    print(f"Output shape:           {tuple(logits.shape)}")
    print(f"Total parameters:       {total_parameters:,}")
    print(f"Trainable parameters:   {trainable_parameters:,}")
    print(
        "Trainable percentage:  "
        f"{100.0 * trainable_parameters / total_parameters:.4f}%"
    )
    print(f"Normalization mean:     {normalization[0]}")
    print(f"Normalization std:      {normalization[1]}")
    print("Result:                 PASSED")

    del dummy_images
    del logits
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Validation device: {device}")

    passed = 0
    failures = []

    for experiment in EXPERIMENTS:
        try:
            validate_model(
                experiment=experiment,
                device=device,
            )
            passed += 1
        except Exception as error:
            failures.append(
                (
                    experiment["name"],
                    str(error),
                )
            )

            print(f"Result: FAILED")
            print(f"Error: {error}")

    print("\n" + "=" * 72)
    print("VALIDATION SUMMARY")
    print("=" * 72)
    print(f"Passed: {passed}/{len(EXPERIMENTS)}")

    if failures:
        print("\nFailures:")

        for experiment_name, error_message in failures:
            print(f"- {experiment_name}: {error_message}")

        raise SystemExit(1)

    print("All model construction tests passed.")


if __name__ == "__main__":
    main()
