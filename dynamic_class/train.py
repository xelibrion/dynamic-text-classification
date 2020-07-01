#!/usr/bin/env python
import argparse
from typing import Dict, Callable
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from catalyst.dl.callbacks import (
    CriterionCallback,
    EarlyStoppingCallback,
    OptimizerCallback,
    CheckpointCallback,
)

from dynamic_class.model import PrototypicalTextClassifier
from dynamic_class.experiment import Experiment
from dynamic_class import utils
from dynamic_class.runner import CustomRunner


logger = utils.get_logger(__name__)


def get_callbacks(config: Dict):
    return [
        CriterionCallback(**config["criterion_callback_params"]),
        OptimizerCallback(**config["optimizer_callback_params"]),
        CheckpointCallback(save_n_best=3),
        EarlyStoppingCallback(**config["early_stopping"]),
    ]


SEED = 42


def get_logdir(config: Dict) -> str:
    baselogdir = Path(__file__).parent / "logs"
    timestamp = utils.get_utcnow_time()
    config_hash = utils.get_short_hash(config)
    logdir = f"{timestamp}.{config_hash}"
    return str(baselogdir / logdir)


def define_args():
    parser = argparse.ArgumentParser(description="DL training pipeline")
    parser.add_argument(
        "--config-path",
        default=Path(__file__).parent / "config.yml",
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--data-path",
        default=Path(__file__).parent / "input",
        help="Folder to load experiment data from",
    )
    return parser


def setup_experiment(args, get_logdir: Callable):
    utils.set_global_seed(SEED)
    utils.prepare_cudnn(deterministic=True, benchmark=False)

    config = utils.load_config(args.config_path, ordered=True)
    logdir = get_logdir(config)

    utils.dump_environment(config, logdir, [args.config_path])
    utils.dump_code(str(Path(__file__).parent), logdir)

    return config, logdir


def load_embeddings(vocab, embeddings_path="./vocab_vectors.txt"):
    print("Loading embeddings")
    embeddings = {}
    with open(embeddings_path) as in_f:
        for line in tqdm(in_f):
            parts = line.split()
            word, numbers = parts[0], parts[1:]
            if word in vocab.word2idx:
                embeddings[word] = np.array([float(x) for x in numbers])
    any_value = next(iter(embeddings.values()))
    matrix = np.zeros((len(embeddings), any_value.shape[0]))
    for idx, w in enumerate(vocab.idx2word):
        if w not in embeddings:
            print(w)
            continue
        matrix[idx] = embeddings[w]
    return matrix


def main(args):
    config, logdir = setup_experiment(args, get_logdir)

    experiment = Experiment(args.data_path, **config["data_params"])

    embeddings = load_embeddings(experiment.vocab)
    print(f"Loaded {len(embeddings)} entries to embeddings")

    model = PrototypicalTextClassifier(
        vocab_size=experiment.vocab_size,
        padding_idx=experiment.vocab.pad,
        pretrained_embeddings=torch.Tensor(embeddings),
        **config["model_params"],
    )
    logger.info(f"The model has {utils.count_parameters(model)} trainable parameters")

    runner = CustomRunner()
    criterion = CrossEntropyLoss()
    optimizer = Adam(
        (p for p in model.parameters() if p.requires_grad), **config["optimizer_params"]
    )
    scheduler = ReduceLROnPlateau(optimizer, **config["scheduler_params"])

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=experiment.get_loaders(),
        callbacks=get_callbacks(config),
        logdir=logdir,
        num_epochs=100,
        main_metric="acc",
        minimize_metric=False,
        verbose=True,
    )


if __name__ == "__main__":
    args = define_args().parse_args()
    main(args)
