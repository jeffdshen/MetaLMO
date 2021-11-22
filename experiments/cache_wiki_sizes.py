from pathlib import Path
import argparse

import numpy as np
from datasets import load_dataset

from data.datasets import WikiDataset


def add_args(parser):
    parser.add_argument(
        "--data_path",
        type=str,
        default="wikipedia",
        help="Huggingface dataset path",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="20200501.en",
        help="Huggingface dataset path",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="save/data/wiki",
        help="Where to save the trained tokenizer files",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=1000,
        help="Validation size.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed")


def main():
    parser = argparse.ArgumentParser("Cache the sizes for wikipedia")
    add_args(parser)
    args = parser.parse_args()

    path = Path(args.save_dir)
    path.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.data_path, args.data_name)

    # NOTE: Don't cache the validation set, because it depennds on the tokenizer.
    dataset = dataset["train"].train_test_split(test_size=args.val_size, seed=args.seed)
    train_dataset = WikiDataset(dataset, "train", "text", tokenizer=None)
    np.save(path / "cached-sizes-train.npy", train_dataset.sizes)


if __name__ == "__main__":
    main()
