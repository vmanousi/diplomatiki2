import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract a backbone-only state dict from a per-epoch DINO "
            "training checkpoint (dino_epoch_XXX.pt), in the same format "
            "as the student_backbone_final.pt / teacher_backbone_final.pt "
            "files pretrain_dino.py saves at the end of a run. Lets you "
            "evaluate an intermediate pretraining epoch's backbone "
            "downstream, the same way the final epoch's backbone already "
            "gets evaluated."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a dino_epoch_XXX.pt checkpoint.",
    )

    parser.add_argument(
        "--which",
        type=str,
        choices=["student", "teacher"],
        default="student",
        help="Which network's backbone to extract (default: student).",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Where to save the extracted backbone-only state dict.",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    state_dict_key = f"{args.which}_state_dict"

    if state_dict_key not in checkpoint:
        raise KeyError(
            f"'{state_dict_key}' not found in checkpoint. "
            f"Available keys: {sorted(checkpoint.keys())}"
        )

    full_state_dict = checkpoint[state_dict_key]

    backbone_state_dict = {
        key[len("backbone."):]: value
        for key, value in full_state_dict.items()
        if key.startswith("backbone.")
    }

    if not backbone_state_dict:
        raise RuntimeError(
            "No 'backbone.*' keys found in "
            f"'{state_dict_key}' — is this really a DINO "
            "training checkpoint?"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(backbone_state_dict, output_path)

    print(f"Source checkpoint: {checkpoint_path}")
    print(f"Source epoch: {checkpoint.get('epoch')}")
    print(f"Extracted: {args.which} backbone")
    print(f"Backbone parameter tensors: {len(backbone_state_dict)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
