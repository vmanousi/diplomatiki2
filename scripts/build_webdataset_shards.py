import argparse
from pathlib import Path

import webdataset as wds
from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Package a flat folder of images into webdataset .tar shards "
            "(<key>.jpg + <key>.cls per sample), in the same format "
            "build_gastrovision_webdataset() expects. The .cls value is a "
            "placeholder (0) — DINO pretraining never reads it, this is "
            "only for datasets with no real labels to package, e.g. a "
            "curated/deduplicated unlabeled pretraining set."
        )
    )

    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Flat folder of input images.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where to write the shard-NNNNNN.tar files.",
    )

    parser.add_argument(
        "--shard-prefix",
        type=str,
        default="train",
        help="Shard filename prefix, e.g. 'train' -> train-000000.tar.",
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=800,
        help="Approximate number of images per shard.",
    )

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in: {images_dir}"
        )

    print(f"Found {len(image_paths)} images in {images_dir}")

    num_shards = max(
        1,
        (len(image_paths) + args.shard_size - 1) // args.shard_size,
    )

    print(f"Writing {num_shards} shard(s) to {output_dir}")

    skipped = 0

    for shard_index in range(num_shards):
        shard_path = (
            output_dir
            / f"{args.shard_prefix}-{shard_index:06d}.tar"
        )

        shard_paths = image_paths[
            shard_index * args.shard_size:
            (shard_index + 1) * args.shard_size
        ]

        with wds.TarWriter(str(shard_path)) as sink:
            for image_path in tqdm(
                shard_paths,
                desc=f"shard {shard_index}",
            ):
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as error:
                    print(
                        f"Skipping unreadable image {image_path}: {error}"
                    )
                    skipped += 1
                    continue

                sink.write(
                    {
                        "__key__": image_path.stem,
                        "jpg": image,
                        "cls": "0",
                    }
                )

    print(f"Done. Skipped {skipped} unreadable image(s).")


if __name__ == "__main__":
    main()
