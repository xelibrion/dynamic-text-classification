from typing import List, Callable, Mapping

import torch
from torch.utils.data import Dataset

from dynamic_class.utils import get_logger
from dynamic_class.vocab import Vocab

logger = get_logger(__name__)


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[str] = None,
        *,
        label_dict: Mapping[str, int] = None,
        vocab: Vocab = None,
        tokenize: Callable = None,
        max_words: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.label_dict = label_dict
        self.vocab = vocab
        self.tokenize = tokenize
        self.max_words = max_words

        if self.label_dict is None and labels is not None:
            # {'class1': 0, 'class2': 1, 'class3': 2, ...}
            self.label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index) -> Mapping[str, torch.Tensor]:

        text = self.texts[index]
        text = " ".join(self.tokenize(text)[: self.max_words])

        word_ids = [self.vocab[w] for w in self.tokenize(text)]
        output_dict = {"inputs": word_ids, "lengths": len(word_ids)}

        if self.labels is not None:
            y = self.labels[index]
            y_encoded = torch.Tensor([self.label_dict.get(y, -1)]).long().squeeze(0)
            output_dict["targets"] = y_encoded

        return output_dict


__all__ = ["TextClassificationDataset"]
