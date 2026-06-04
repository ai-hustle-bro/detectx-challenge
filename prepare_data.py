"""Prepare folder-per-class image datasets for the DetectX competition layout."""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {
    ".jpeg",
    ".jpg",
    ".png",
}


def parse_args():
    """Parse command-line arguments for the dataset preparation utility."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert a folder-per-class image dataset into the DetectX "
            "competition data layout."
        )
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory containing one subdirectory per class.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where train/test folders and CSV files will be written.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of each class to place in the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the deterministic train/test split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def find_class_dirs(source_dir):
    """Return sorted class directories from a folder-per-class dataset."""
    if not source_dir.exists() or not source_dir.is_dir():
        raise ValueError(f"source_dir is not a directory: {source_dir}")

    class_dirs = sorted(path for path in source_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found in {source_dir}")
    return class_dirs


def find_images(class_dir):
    """Return supported image files under a class directory."""
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_images(images, test_ratio, rng):
    """Split one class of images into deterministic train and test subsets."""
    shuffled = list(images)
    rng.shuffle(shuffled)

    if not 0 <= test_ratio < 1:
        raise ValueError("--test-ratio must be >= 0 and < 1")

    test_count = int(round(len(shuffled) * test_ratio))
    if test_ratio > 0 and len(shuffled) > 1:
        test_count = max(1, min(test_count, len(shuffled) - 1))

    test_images = shuffled[:test_count]
    train_images = shuffled[test_count:]
    return train_images, test_images


def reset_output_dir(output_dir, overwrite):
    """Create fresh train/test output directories."""
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    (output_dir / "train").mkdir(parents=True)
    (output_dir / "test").mkdir(parents=True)


def copy_split(images, split_dir, split_name, label_id, start_index):
    """Copy split images and return CSV label rows plus the next image index."""
    rows = []
    index = start_index
    for image_path in images:
        image_id = f"{split_name}_{index:06d}{image_path.suffix.lower()}"
        shutil.copy2(image_path, split_dir / image_id)
        rows.append({"image_id": image_id, "label": label_id})
        index += 1
    return rows, index


def write_labels(path, rows):
    """Write image label rows using the competition CSV schema."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "label"])
        writer.writeheader()
        writer.writerows(rows)


def prepare_dataset(source_dir, output_dir, test_ratio, seed, overwrite):
    """Convert a folder-per-class dataset into the DetectX competition format."""
    class_dirs = find_class_dirs(source_dir)
    reset_output_dir(output_dir, overwrite)

    rng = random.Random(seed)
    label_map = {}
    train_rows = []
    test_rows = []
    train_index = 1
    test_index = 1

    for label_id, class_dir in enumerate(class_dirs):
        images = find_images(class_dir)
        if not images:
            raise ValueError(f"No supported images found in class directory: {class_dir}")

        label_map[label_id] = class_dir.name
        train_images, test_images = split_images(images, test_ratio, rng)

        rows, train_index = copy_split(
            train_images,
            output_dir / "train",
            "train",
            label_id,
            train_index,
        )
        train_rows.extend(rows)

        rows, test_index = copy_split(
            test_images,
            output_dir / "test",
            "test",
            label_id,
            test_index,
        )
        test_rows.extend(rows)

    write_labels(output_dir / "labels_train.csv", train_rows)
    write_labels(output_dir / "secret_ground_truth.csv", test_rows)

    with (output_dir / "label_map.json").open("w", encoding="utf-8") as handle:
        json.dump(label_map, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return {
        "classes": len(label_map),
        "train_images": len(train_rows),
        "test_images": len(test_rows),
        "label_map": label_map,
    }


def main():
    """Run the dataset preparation command-line workflow."""
    args = parse_args()
    summary = prepare_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    print(
        "Prepared dataset: "
        f"{summary['classes']} classes, "
        f"{summary['train_images']} train images, "
        f"{summary['test_images']} test images"
    )
    print(f"Wrote output to: {args.output_dir}")


if __name__ == "__main__":
    main()
