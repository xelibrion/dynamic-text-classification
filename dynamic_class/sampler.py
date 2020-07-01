from typing import Tuple, Iterator, Dict, Any, Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class BaseSampler(object):
    def __init__(
        self,
        data: Sequence[Tuple[Tensor, Tensor]],
        shuffle: bool = False,
        pad_index: int = 0,
        batch_size: int = 128,
    ):
        """A basic sampler.

        Parameters
        ----------
        data : Sequence[Tuple[Tensor, Tensor]]
            The input data to sample from, as a list of
            (source, target) pairs
        shuffle : bool, optional
            Whether to shuffle the data, by default True
        pad_index : int, optional
            The padding index, by default 0
        batch_size : int, optional
            The batch size to use, by default 128

        """
        self.data = data
        self.shuffle = shuffle
        self.pad = pad_index
        self.batch_size = batch_size

    def __iter__(self):
        """Sample from the list of features and yields batches.

        Yields
        ------
        Iterator[Tuple[Tensor, Tensor]]
            In order: source, target
            For sequences, the batch is used as first dimension.

        """
        if self.shuffle:
            indices = np.random.permutation(len(self.data))
        else:
            indices = list(range(len(self.data)))

        num_batches = len(indices) // self.batch_size
        indices_splits = np.array_split(indices, num_batches)
        for split in indices_splits:
            examples = [self.data[i] for i in split]
            source, target = list(zip(*examples))
            source = pad_sequence(source, batch_first=True, padding_value=self.pad)
            target = torch.tensor(target)
            yield (source.long(), target.long())
