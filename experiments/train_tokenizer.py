from pathlib import Path
import argparse

from datasets import load_dataset
import tokenizers

from data.tokenizers import BatchedTextIterable


def add_args(parser):
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=30000,
        help="Vocab size.",
    )
    parser.add_argument(
        "--cls_count",
        type=int,
        default=30,
        help="Number of different cls tokens.",
    )
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
        default="save/tokenizers/wiki-bpe",
        help="Where to save the trained tokenizer files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="How many texts to load a time",
    )


def main():
    parser = argparse.ArgumentParser("Use wikipedia to train a BPE Tokenizer")
    add_args(parser)
    args = parser.parse_args()

    path = Path(args.save_dir)
    path.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.data_path, args.data_name)
    dataset_iterable = BatchedTextIterable(dataset, "train", "text", args.batch_size)
    tokenizer = tokenizers.ByteLevelBPETokenizer()
    special_tokens = ["[PAD]", "[MASK]", "[SEP]"]
    for i in range(args.cls_count):
        special_tokens.append("[CLS{}]".format(i))

    tokenizer.train_from_iterator(
        dataset_iterable, vocab_size=args.vocab_size, special_tokens=special_tokens
    )

    tokenizer.save_model(args.save_dir)


if __name__ == "__main__":
    main()
