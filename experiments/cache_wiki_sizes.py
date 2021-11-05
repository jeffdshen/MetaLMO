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


def main():
    parser = argparse.ArgumentParser("Use wikipedia to train a BPE Tokenizer")
    add_args(parser)
    args = parser.parse_args()

    path = Path(args.save_dir)
    path.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.data_path, args.data_name)
    wiki_dataset = WikiDataset(dataset, "train", "text", None)

    np.save(path / "cached-sizes", wiki_dataset.sizes)


if __name__ == "__main__":
    main()
