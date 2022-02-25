import random
import re
from itertools import groupby

import numpy as np
import torch
import torch.nn.functional as F

from tokenizers import Tokenizer

from .metrics import metric_accuracy, metric_f1, metric_multirc, metric_record


class Labels:
    def __init__(self, labels):
        self.label_to_token = {raw: token for raw, token in labels}
        self.labels, self.classes = zip(*labels)

    def to_token(self, label):
        return self.label_to_token[label]

    def from_index(self, index):
        return self.labels[index]

    def all_tokens(self):
        return self.classes


def select_encodings(encoding, window):
    valid_windows = ["first", "all", "random"]
    if window not in valid_windows:
        raise ValueError("window must be one of {}".format(valid_windows))

    if window == "first":
        return [encoding]
    elif window == "all":
        all = [encoding] + encoding.overflowing
        return all
    elif window == "random":
        choice = random.choice([encoding] + encoding.overflowing)
        return [choice]


def selected_encodings_to_features(encoding_list):
    ids = [encoding.ids for encoding in encoding_list]
    return torch.tensor(ids, dtype=torch.long)


def encoding_to_features(encoding, window):
    return selected_encodings_to_features(select_encodings(encoding, window))


def prepare_task(task_id, pad_id, example, encoding, labels: Labels, window: str):
    features = encoding_to_features(encoding, window)
    features[:, 0] = task_id
    token = labels.to_token(example["label"]) if "label" in example else 0
    labels = torch.full_like(features, token, dtype=torch.long)
    labels.masked_fill_(features == pad_id, pad_id)
    idxs = torch.full((features.size(0),), example["idx"])
    return idxs, features, labels


def predict_argmax_mean(idxs, inputs, outputs, pad_id, labels: Labels):
    # set padding outputs to 0
    padding_mask = (inputs == pad_id).unsqueeze(-1)
    outputs = outputs.masked_fill(padding_mask, 0.0)

    # take the mean of each class across the sequence
    classes = torch.tensor(labels.all_tokens(), dtype=torch.long, device=outputs.device)
    outputs = outputs.index_select(-1, classes)
    outputs = torch.mean(outputs, dim=-2)

    # take the argmax of mean across all windows
    unique_idxs, counts = idxs.unique_consecutive(dim=0, return_counts=True)
    outputs = outputs.split(counts.tolist())
    outputs = tuple(torch.argmax(torch.mean(y, dim=0), dim=0).item() for y in outputs)
    return {
        idx.item(): labels.from_index(output)
        for idx, output in zip(unique_idxs, outputs)
    }


def predict_mlm(idxs, inputs, outputs, pad_id):
    outputs = torch.argmax(outputs, dim=-1)
    padding_mask = inputs == pad_id
    outputs = outputs.masked_fill(padding_mask, pad_id).tolist()

    return {idx.item(): output for idx, output in zip(idxs, outputs)}


def split_list(l, delim_pred):
    return [list(g) for k, g in groupby(l, delim_pred) if k]


def get_max_value_span(spans):
    if not spans:
        return []
    span = max(spans, key=lambda span: max(span, key=lambda x: x[1])[1])
    span, _ = list(zip(*span))
    return span


def predict_span(idxs, inputs, outputs, pad_id, none_label, span_reduce):
    # Take the max, argmax of the log softmax
    outputs = F.log_softmax(outputs, dim=-1)
    values, outputs = torch.max(outputs, dim=-1)

    # set padding outputs to none_label
    padding_mask = inputs == pad_id
    outputs = outputs.masked_fill(padding_mask, none_label)

    unique_idxs, counts = idxs.unique_consecutive(dim=0, return_counts=True)
    values, outputs = values.split(counts.tolist()), outputs.split(counts.tolist())

    # 2, N, W, S -> N, (2, W, S) -> N, W, (2, S) -> N, W, S, 2
    output_values = [
        [list(zip(w, z)) for w, z in zip(o, v)] for o, v in zip(outputs, values)
    ]
    delim = lambda x: x[0] != none_label
    spans = []
    for output in output_values:
        all_spans = []
        for window in output:
            all_spans += split_list(window, delim)
        span = span_reduce(all_spans)
        spans.append(span)

    return {idx.item(): span for idx, span in zip(unique_idxs, spans)}


def str_insert(s, indices, t):
    indices = [0] + sorted(indices)
    splits = [s[a:b] for a, b in zip(indices[:-1], indices[1:])] + [s[indices[-1] :]]
    return t.join(splits)


class BoolQTask:
    def __init__(self, task_id, tokenizer, labels):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.padding["pad_id"]
        self.task_id = task_id
        self.labels = Labels(labels)

    def encode(self, example, window):
        encoding = self.tokenizer.encode(example["question"], example["passage"])
        return prepare_task(
            self.task_id, self.pad_id, example, encoding, self.labels, window
        )

    def predict(self, idxs, inputs, outputs):
        return predict_argmax_mean(idxs, inputs, outputs, self.pad_id, self.labels)

    def score(self, preds, examples, strict):
        return metric_accuracy(preds, examples, strict)


class CBTask:
    def __init__(self, task_id, tokenizer, labels):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.padding["pad_id"]
        self.task_id = task_id
        self.labels = Labels(labels)

    def encode(self, example, window):
        encoding = self.tokenizer.encode(example["hypothesis"], example["premise"])
        return prepare_task(
            self.task_id, self.pad_id, example, encoding, self.labels, window
        )

    def predict(self, idxs, inputs, outputs):
        return predict_argmax_mean(idxs, inputs, outputs, self.pad_id, self.labels)

    def score(self, preds, examples, strict):
        return (
            metric_f1(preds, examples, strict),
            metric_accuracy(preds, examples, strict),
        )


class COPATask:
    def __init__(self, task_id, tokenizer, labels):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.padding["pad_id"]
        self.task_id = task_id
        self.labels = Labels(labels)

    def encode(self, example, window):
        question = example["premise"] + " " + example["question"]
        choices = "0: " + example["choice1"] + " 1: " + example["choice2"]
        encoding = self.tokenizer.encode(question, choices)
        return prepare_task(
            self.task_id, self.pad_id, example, encoding, self.labels, window
        )

    def predict(self, idxs, inputs, outputs):
        return predict_argmax_mean(idxs, inputs, outputs, self.pad_id, self.labels)

    def score(self, preds, examples, strict):
        return metric_accuracy(preds, examples, strict)


class RTETask:
    def __init__(self, task_id, tokenizer, labels):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.padding["pad_id"]
        self.task_id = task_id
        self.labels = Labels(labels)

    def encode(self, example, window):
        encoding = self.tokenizer.encode(example["hypothesis"], example["premise"])
        return prepare_task(
            self.task_id, self.pad_id, example, encoding, self.labels, window
        )

    def predict(self, idxs, inputs, outputs):
        return predict_argmax_mean(idxs, inputs, outputs, self.pad_id, self.labels)

    def score(self, preds, examples, strict):
        return metric_accuracy(preds, examples, strict)


class WiCTask:
    def __init__(self, task_id, tokenizer, labels):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.padding["pad_id"]
        self.task_id = task_id
        self.labels = Labels(labels)

    def encode(self, example, window):
        s1, s2 = example["sentence1"], example["sentence2"]
        a1, b1, a2, b2 = (
            example["start1"],
            example["end1"],
            example["start2"],
            example["end2"],
        )

        s1 = str_insert(s1, [a1, b1], "*")
        s2 = str_insert(s2, [a2, b2], "*")
        encoding = self.tokenizer.encode(s1, s2)
        return prepare_task(
            self.task_id, self.pad_id, example, encoding, self.labels, window
        )

    def predict(self, idxs, inputs, outputs):
        return predict_argmax_mean(idxs, inputs, outputs, self.pad_id, self.labels)

    def score(self, preds, examples, strict):
        return metric_accuracy(preds, examples, strict)


def indexed_split(s):
    return [(m.start(), m.group()) for m in re.finditer(r"\S+", s)]


def to_lower_alnum(s):
    return re.sub(r"\W+", "", s).lower()


def get_wsc_span(s, t, index):
    split = indexed_split(s)
    start = split[index][0]
    if s[start:].lower().startswith(t.lower()):
        return (start, start + len(t))

    # Handle buggy examples: select span with the most words in t
    words = set(to_lower_alnum(x) for x in t.split())
    span = []
    for i in range(index, len(split)):
        if to_lower_alnum(split[i][1]) in words:
            span.append(i)
            words.remove(to_lower_alnum(split[i][1]))

    if len(span) > 0:
        i, j = span[0], span[-1]
        return split[i][0], split[j][0] + len(split[j][1])
    return 0, 0


def overlap(a, b, c, d):
    return a < d and c < b


class WSCTask:
    def __init__(self, task_id, tokenizer, labels):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.padding["pad_id"]
        self.task_id = task_id
        self.labels = Labels(labels)

    def encode(self, example, window):
        text = example["text"]
        target = example["target"]
        start1, end1 = get_wsc_span(text, target["span1_text"], target["span1_index"])
        start2, end2 = get_wsc_span(text, target["span2_text"], target["span2_index"])
        text = str_insert(text, [start1, end1, start2, end2], "*")
        question = "{} = {}".format(target["span1_text"], target["span2_text"])

        encoding = self.tokenizer.encode(question, text)
        return prepare_task(
            self.task_id, self.pad_id, example, encoding, self.labels, window
        )

    def predict(self, idxs, inputs, outputs):
        return predict_argmax_mean(idxs, inputs, outputs, self.pad_id, self.labels)

    def score(self, preds, examples, strict):
        return metric_accuracy(preds, examples, strict)


class ReCoRDTask:
    def __init__(self, task_id, tokenizer, none_label):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.padding["pad_id"]
        self.task_id = task_id
        self.none_label = none_label

    def _make_labels(self, encodings, answers, starts, ends):
        all_indices = sorted(starts + ends)
        index_map = {x: x + i for i, x in enumerate(all_indices)}
        answer_map = set()
        for answer in answers:
            start, end = index_map[answer["start"]] + 1, index_map[answer["end"]] + 1
            for i in range(start, end):
                answer_map.add(i)

        labels_batch = []
        for e in encodings:
            labels = []
            for id, type_id, (start, end) in zip(e.ids, e.type_ids, e.offsets):
                label = type_id == 1 and any(
                    (i in answer_map) for i in range(start, end)
                )
                label = id if label else self.none_label
                labels.append(label)
            labels_batch.append(labels)
        labels = torch.tensor(labels_batch, dtype=torch.long)
        return labels

    def encode(self, example, window):
        passage = example["passage"]
        text = passage["text"]
        qa = example["qas"]
        starts, ends = zip(*((e["start"], e["end"]) for e in passage["entities"]))
        text = str_insert(text, (starts + tuple(e + 1 for e in ends)), "*")
        encoding = self.tokenizer.encode(qa["query"], text)

        encodings = select_encodings(encoding, window)
        features = selected_encodings_to_features(encodings)
        features[:, 0] = self.task_id
        if "answers" in qa:
            labels = self._make_labels(encodings, qa["answers"], starts, ends)
            labels.masked_fill_(features == self.pad_id, self.pad_id)
        else:
            labels = torch.zeros_like(features, dtype=torch.long)
        idxs = torch.full((features.size(0),), example["qas"]["idx"])
        return idxs, features, labels

    def predict(self, idxs, inputs, outputs):
        spans = predict_span(
            idxs, inputs, outputs, self.pad_id, self.none_label, get_max_value_span
        )
        return {idx: self.tokenizer.decode(span) for idx, span in spans.items()}

    def score(self, preds, examples, strict):
        return metric_record(preds, examples, strict)


class MultiRCTask:
    def __init__(self, task_id, tokenizer, labels):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.padding["pad_id"]
        self.task_id = task_id
        self.labels = Labels(labels)

    def encode(self, example, window):
        text = example["passage"]["text"]
        questions = example["passage"]["questions"]
        answers = questions["answers"]
        question = questions["question"]
        answer = answers["text"]
        label = answers["label"] if "label" in answers else None
        idx = answers["idx"]
        full_question = "Question: {} Answer: {}".format(question, answer)
        encoding = self.tokenizer.encode(full_question, text)
        features = encoding_to_features(encoding, window)
        features[:, 0] = self.task_id
        token = self.labels.to_token(label) if label is not None else 0
        labels = torch.full_like(features, token, dtype=torch.long)
        labels.masked_fill_(features == self.pad_id, self.pad_id)
        idxs = torch.full((features.size(0),), idx)
        return idxs, features, labels

    def predict(self, idxs, inputs, outputs):
        return predict_argmax_mean(idxs, inputs, outputs, self.pad_id, self.labels)

    def score(self, preds, examples, strict):
        return metric_multirc(preds, examples, strict)


class WhichMoonTask:
    def __init__(self, task_id, tokenizer, labels):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.padding["pad_id"]
        self.task_id = task_id
        self.labels = Labels(labels)

    def encode(self, example, window):
        encoding = self.tokenizer.encode(example["question"])
        return prepare_task(
            self.task_id, self.pad_id, example, encoding, self.labels, window
        )

    def predict(self, idxs, inputs, outputs):
        return predict_argmax_mean(idxs, inputs, outputs, self.pad_id, self.labels)

    def score(self, preds, examples, strict):
        return metric_accuracy(preds, examples, strict)


def get_random_weights(max_tokens, special_tokens):
    random_weights = [1] * max_tokens
    for i in special_tokens:
        if i >= 0 and i < len(random_weights):
            random_weights[i] = 0
    return random_weights


class MLMTask:
    def __init__(
        self,
        task_id,
        tokenizer,
        mask_id,
        random_weights,
        mask_prob,
        unmask_prob,
        randomize_prob,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.padding["pad_id"]
        self.task_id = task_id
        self.mask_id = mask_id
        self.mask_prob = mask_prob
        self.unmask_prob = unmask_prob
        self.randomize_prob = randomize_prob
        self.random_weights = torch.tensor(random_weights, dtype=torch.float)

    def mask1d_(self, x):
        x = x.detach().clone()
        x[0] = self.task_id
        y = x.detach().clone()
        size = (x != self.pad_id).sum().item()
        num_mask = int(self.mask_prob * size + random.random())

        masks = torch.tensor(random.sample(range(size), num_mask), dtype=torch.long)
        change_masks = torch.rand(num_mask)
        unmask = change_masks < self.unmask_prob
        random_mask = change_masks < (self.randomize_prob + self.unmask_prob)
        random_mask = random_mask & (~unmask)
        if random_mask.sum().item() > 0:
            random_content = torch.multinomial(
                self.random_weights, random_mask.sum().item(), replacement=True
            )
        else:
            random_content = torch.tensor([], dtype=torch.long)

        masked = torch.full_like(x, False, dtype=torch.bool)
        masked[masks] = True

        x[masks[~unmask]] = self.mask_id
        x[masks[random_mask]] = random_content
        y[~masked] = self.pad_id
        x[0] = self.task_id
        return x, y

    def encode(self, example, window):
        idx, example = example
        features, labels = self.mask1d_(example)
        features = features.unsqueeze(0)
        labels = labels.unsqueeze(0)
        return idx, features, labels

    def predict(self, idxs, inputs, outputs):
        return predict_mlm(idxs, inputs, outputs, self.pad_id)

    def score_single(self, idxs, preds, labels, strict):
        scores = {}
        for i, idx in enumerate(idxs.tolist()):
            label = labels[i]

            if not strict and idx not in preds:
                continue

            pred = preds[idx]
            correct = sum([l != self.pad_id and l == p for l, p in zip(label, pred)])
            total = sum([l != self.pad_id for l in label])
            if total > 0:
                scores[idx] = float(correct / total)
            else:
                scores[idx] = 1.0
        return scores

    def score(self, preds, examples, strict):
        raise NotImplementedError("Can only score single")


def get_mlm_task(
    tokenizer: Tokenizer, mask_prob: float, unmask_prob: float, randomize_prob: float
) -> MLMTask:
    task_id = tokenizer.token_to_id("[CLS1]")
    mask_id = tokenizer.token_to_id("[MASK]")
    pad_id = tokenizer.token_to_id("[PAD]")
    max_tokens = tokenizer.get_vocab_size()

    # sep_id is ok to generate for mlm
    special_tokens = [pad_id, mask_id]
    for i in range(max_tokens):
        token = tokenizer.token_to_id("[CLS{}]".format(i))
        if token is None:
            break
        special_tokens.append(token)

    return MLMTask(
        task_id,
        tokenizer,
        mask_id,
        get_random_weights(max_tokens, special_tokens),
        mask_prob,
        unmask_prob,
        randomize_prob,
    )
