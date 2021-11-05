import pathlib

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing


class TextIterable:
    def __init__(self, dataset, split, column):
        self.dataset = dataset
        self.split = split
        self.column = column

    def __iter__(self):
        return map(lambda x: self.dataset[self.split][x][self.column], range(len(self)))

    def __len__(self):
        return len(self.dataset[self.split])


class BatchedTextIterable:
    def __init__(self, dataset, split, column, batch_size):
        self.dataset = dataset
        self.split = split
        self.column = column
        self.batch_size = batch_size

    def __iter__(self):
        return map(
            lambda x: self.dataset[self.split][x : x + self.batch_size][self.column],
            range(0, len(self), self.batch_size),
        )

    def __len__(self):
        return len(self.dataset[self.split])


def get_task_tokenizer(path, model_max_length, stride, **kwargs):
    vocab_file = str(pathlib.Path(path, "vocab.json"))
    merges_file = str(pathlib.Path(path, "merges.txt"))

    tokenizer = ByteLevelBPETokenizer.from_file(vocab_file, merges_file, **kwargs)
    tokenizer.enable_truncation(
        model_max_length, stride=stride, strategy="longest_first"
    )
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"))
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS0] $A [SEP]",
        pair="[CLS0] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS0]", tokenizer.token_to_id("[CLS0]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    return tokenizer


def get_pretrain_tokenizer(path, model_max_length, **kwargs):
    vocab_file = str(pathlib.Path(path, "vocab.json"))
    merges_file = str(pathlib.Path(path, "merges.txt"))

    tokenizer = ByteLevelBPETokenizer.from_file(vocab_file, merges_file, **kwargs)
    tokenizer.enable_truncation(
        model_max_length, stride=0, strategy="only_first"
    )
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"))
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS0] $A [SEP]",
        pair="[CLS0] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS0]", tokenizer.token_to_id("[CLS0]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    return tokenizer


def get_fake_tokenizer(texts, vocab_size, cls_count, model_max_length, stride):
    tokenizer = ByteLevelBPETokenizer()
    special_tokens = ["[PAD]", "[MASK]", "[SEP]"]
    for i in range(cls_count):
        special_tokens.append("[CLS{}]".format(i))
    tokenizer.train_from_iterator(
        texts, vocab_size=vocab_size, special_tokens=special_tokens, show_progress=False
    )

    tokenizer.enable_truncation(
        model_max_length, stride=stride, strategy="longest_first"
    )
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"))
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS0] $A [SEP]",
        pair="[CLS0] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS0]", tokenizer.token_to_id("[CLS0]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    return tokenizer
