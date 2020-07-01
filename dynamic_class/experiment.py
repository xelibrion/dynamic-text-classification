from typing import List, Mapping
from collections import OrderedDict
from pathlib import Path
from functools import partial

import pandas as pd
import torch
from torch.utils import data

from dynamic_class.vocab import Vocab
from dynamic_class.dataset_regular import TextClassificationDataset
from dynamic_class.dataset_episodic import EpisodesDataset
from dynamic_class import utils


logger = utils.get_logger(__name__)


def tokenize(text: str) -> List[str]:
    return [x for x in text.split() if x]


def pad_sequences(pad_id: int, batch: List[Mapping]):
    elem = batch[0]

    for key in ["inputs"]:
        seq_length = [len(x[key]) for x in batch]
        max_length = max(seq_length)
        pad_length = [max_length - x for x in seq_length]
        for idx, item in enumerate(batch):
            item[key] = item[key] + [pad_id] * pad_length[idx]

    return {key: torch.as_tensor([d[key] for d in batch]) for key in elem}


class Experiment:
    def __init__(
        self,
        path_to_data: str,
        *,
        mode: str = "train",
        batch_size: int = None,
        vocab_max_size: int = None,
        train_filename: str = "train.csv",
        val_filename: str = "val.csv",
        **kwargs,
    ):
        self.path_to_data = path_to_data
        self.batch_size = batch_size
        self.train_filename = train_filename
        self.val_filename = val_filename
        self.kwargs = kwargs
        if mode == "train":
            self.vocab = self.build_vocab(vocab_max_size)
            self.vocab.save(Path(path_to_data) / "vocab.txt")
            self.train_dataset, self.val_dataset = self.load_datasets()
        else:
            self.vocab = Vocab.load(Path(path_to_data) / "vocab.txt")

    def build_vocab(self, vocab_max_size: int):
        path_to_data = Path(self.path_to_data)
        train_df = pd.read_csv(path_to_data / self.train_filename)

        logger.info("Building vocabulary")
        vocab_obj, n_unk = Vocab.build(
            train_df["text"], tokenize=tokenize, max_size=vocab_max_size
        )
        logger.info(f"Vocabulary size: {len(vocab_obj)}")
        logger.info(f"# of <unk> token: {n_unk}")
        return vocab_obj

    @property
    def vocab_size(self):
        return len(self.vocab)

    def load_datasets(self):
        path_to_data = Path(self.path_to_data)

        train_df = pd.read_csv(path_to_data / self.train_filename)
        val_df = pd.read_csv(path_to_data / self.val_filename)

        train_dataset = TextClassificationDataset(
            texts=train_df["text"].values.tolist(),
            labels=train_df["target"].values.tolist(),
            vocab=self.vocab,
            tokenize=tokenize,
        )
        num_classes = train_df["target"].nunique()
        print(f"Number of classes: {num_classes}")

        val_dataset = TextClassificationDataset(
            texts=val_df["text"].values.tolist(),
            labels=val_df["target"].values.tolist(),
            label_dict=train_dataset.label_dict,
            vocab=self.vocab,
            tokenize=tokenize,
        )

        return train_dataset, val_dataset

    def get_loaders(self):
        pad_sequences_fn = partial(pad_sequences, self.vocab.pad)

        episodes = EpisodesDataset(
            self.train_dataset,
            n_episodes=self.kwargs["n_episodes"],
            n_support=self.kwargs["n_support"],
            n_query=self.kwargs["n_query"],
            pad_index=self.vocab.pad,
        )
        train_loader = data.DataLoader(episodes,)

        prototypes_loader = data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, collate_fn=pad_sequences_fn,
        )
        val_loader = data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=pad_sequences_fn,
            shuffle=True,
        )
        return OrderedDict(
            train=train_loader, prototypes=prototypes_loader, valid=val_loader
        )

    def get_loader(self, data_path: str):
        logger.info(f"Loading dataset from '{data_path}'")
        df = pd.read_csv(data_path)

        dataset = TextClassificationDataset(
            texts=df["text"].values.tolist(),
            labels=df["target"].values.tolist() if "target" in df.columns else None,
            vocab=self.vocab,
            tokenize=tokenize,
        )

        pad_sequences_fn = partial(pad_sequences, self.vocab.pad)

        return data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=pad_sequences_fn,
        )
