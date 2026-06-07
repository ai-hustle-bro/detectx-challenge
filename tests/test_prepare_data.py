"""Tests for the folder-per-class dataset preparation utility."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from prepare_data import prepare_dataset


class PrepareDatasetTests(unittest.TestCase):
    """Cover the generated competition files from a small source dataset."""

    def test_prepare_dataset_writes_expected_files_and_label_map(self):
        """Convert a two-class dataset and verify the organizer-facing outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "raw"
            output_dir = root / "data"

            for class_name, extension in (("Cats", ".jpg"), ("Dogs", ".png")):
                class_dir = source_dir / class_name
                class_dir.mkdir(parents=True)
                for index in range(2):
                    (class_dir / f"{class_name.lower()}_{index}{extension}").write_text(
                        "image",
                        encoding="utf-8",
                    )

            summary = prepare_dataset(
                source_dir=source_dir,
                output_dir=output_dir,
                test_ratio=0.5,
                seed=123,
                overwrite=False,
            )

            self.assertEqual(
                summary,
                {
                    "classes": 2,
                    "train_images": 2,
                    "test_images": 2,
                    "label_map": {0: "Cats", 1: "Dogs"},
                },
            )

            with (output_dir / "label_map.json").open(encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {"0": "Cats", "1": "Dogs"})

            with (output_dir / "labels_train.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                train_rows = list(csv.DictReader(handle))
            with (output_dir / "secret_ground_truth.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                test_rows = list(csv.DictReader(handle))

            self.assertEqual([row["label"] for row in train_rows], ["0", "1"])
            self.assertEqual([row["label"] for row in test_rows], ["0", "1"])
            self.assertEqual(len(list((output_dir / "train").iterdir())), 2)
            self.assertEqual(len(list((output_dir / "test").iterdir())), 2)


if __name__ == "__main__":
    unittest.main()
