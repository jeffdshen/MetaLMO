# Copyright (c) Jeffrey Shen

import argparse

from trainers import (
    finetuner,
    meta_pretrainer,
    roberta_pretrainer,
    predicter,
    soft_pretrainer,
)


def add_subparser(name, subparsers, parent_parser, module):
    subparser = subparsers.add_parser(
        name,
        parents=[parent_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    module.add_train_args(subparser)
    subparser.set_defaults(train=module.train)


def main(args=None):
    parser = argparse.ArgumentParser("Run a model")
    parent = argparse.ArgumentParser(add_help=False)

    subparsers = parser.add_subparsers()

    add_subparser("meta_pretrain", subparsers, parent, meta_pretrainer)
    add_subparser("soft_pretrain", subparsers, parent, soft_pretrainer)
    add_subparser("roberta_pretrain", subparsers, parent, roberta_pretrainer)
    add_subparser("finetune", subparsers, parent, finetuner)
    add_subparser("predict", subparsers, parent, predicter)

    args = parser.parse_args(args)

    train = args.train
    del args.train
    train(args)


if __name__ == "__main__":
    main()
