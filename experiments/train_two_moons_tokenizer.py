from pathlib import Path
import argparse

from data.config import two_moons
import tokenizers


def add_args(parser):
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=512,
        help="Vocab size.",
    )
    parser.add_argument(
        "--cls_count",
        type=int,
        default=30,
        help="Number of different cls tokens.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="save/tokenizers/two-moons",
        help="Where to save the trained tokenizer files",
    )


def main():
    parser = argparse.ArgumentParser("Use two moons to train a BPE Tokenizer")
    add_args(parser)
    args = parser.parse_args()

    path = Path(args.save_dir)
    path.mkdir(parents=True, exist_ok=True)
    dataset_iterable = two_moons.get_raw_data()["TWO_MOONS"]
    tokenizer = tokenizers.ByteLevelBPETokenizer()
    special_tokens = ["[PAD]", "[MASK]", "[SEP]"]
    for i in range(args.cls_count):
        special_tokens.append("[CLS{}]".format(i))

    # Ensure every token compiles to one id
    for i in range(100):
        dataset_iterable.append(str(i))
    dataset_iterable.append(" ".join(str(i % 100) for i in range(200)))
    tokenizer.train_from_iterator(
        dataset_iterable,
        vocab_size=args.vocab_size,
        min_frequency=1,
        special_tokens=special_tokens,
    )

    tokenizer.save_model(args.save_dir)


if __name__ == "__main__":
    main()
