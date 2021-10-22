from pathlib import Path
import argparse

from datasets import load_dataset
import tokenizers

from data.tokenizers import TextIterable


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


def main():
    parser = argparse.ArgumentParser("Use wikipedia to train a BPE Tokenizer")
    add_args(parser)
    args = parser.parse_args()

    dataset = load_dataset(args.data_path, args.data_name)
    dataset_iterable = TextIterable(dataset, "train", "text")
    tokenizer = tokenizers.ByteLevelBPETokenizer()
    special_tokens = ["[PAD]", "[MASK]", "[SEP]"]
    if args.cls_count == 1:
        special_tokens.append("[CLS]")
    else:
        for i in range(args.cls_count):
            special_tokens.append("[CLS{}]".format(i))

    tokenizer.train_from_iterator(
        dataset_iterable, vocab_size=args.vocab_size, special_tokens=special_tokens
    )

    path = Path()
    path.mkdir(parents=True, exists_ok=True)
    tokenizer.save_model(args.save_dir)


if __name__ == "__main__":
    main()
