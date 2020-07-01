from typing import List, Dict, Callable
from collections import Counter
import logging


class Vocab:
    def __init__(self, word_count: Dict[str, int]):
        self.word_count = word_count
        self.word2idx = {}
        self.idx2word = []

        for w in self.word_count.keys():
            self.word2idx[w] = len(self.word2idx)
            self.idx2word.append(w)

        self.size = len(self.word2idx)

        self.pad = self.word2idx["<pad>"]
        self.go = self.word2idx["<go>"]
        self.eos = self.word2idx["<eos>"]
        self.unk = self.word2idx["<unk>"]
        self.blank = self.word2idx["<blank>"]

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        if key in self.word2idx:
            return self.word2idx[key]
        return self.unk

    @staticmethod
    def build(texts: List[str], *, tokenize: Callable, max_size=25000):
        v = ["<pad>", "<go>", "<eos>", "<unk>", "<blank>"]
        words = [w for item in texts for w in tokenize(item)]
        cnt = Counter(words)
        n_unk = len(words)
        for w, c in cnt.most_common(max_size):
            v.append(w)
            n_unk -= c
        cnt["<unk>"] = n_unk

        return Vocab({w: cnt[w] for w in v}), n_unk

    def save(self, path: str):
        with open(path, "w") as f:
            for w, c in self.word_count.items():
                f.write(f"{w}\t{c}\n")

    @staticmethod
    def load(path: str):
        logging.info(f"Loading vocabulary from {path}")
        word_count = {}
        with open(path) as f:
            for line in f:
                [w, c] = line.strip().split()
                word_count[w] = c

        logging.info(f"Loaded vocabulary with {len(word_count)} words")
        return Vocab(word_count)
