#!/usr/bin/env python
from typing import Dict
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from catalyst.dl import SupervisedRunner
from catalyst.tools.typing import Model

from dynamic_class.runner import CustomRunner
from dynamic_class.model import PrototypicalTextClassifier
from dynamic_class.experiment import Experiment
from dynamic_class import utils


def define_args():
    parser = argparse.ArgumentParser(description="DL predict pipeline")
    parser.add_argument(
        "--config-path",
        default=Path(__file__).parent / "config.yml",
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Model checkpoint to load",
    )
    parser.add_argument(
        "--vocab-path",
        default=Path(__file__).parent / "input/vocab.txt",
        help="Path to vocabulary",
    )
    parser.add_argument(
        "--mode",
        choices=["embeddings", "prototypes"],
        default="embeddings",
        help="Mode to run the script",
    )
    parser.add_argument(
        "dataset", help="Dataset to run predicts on",
    )
    return parser


SEED = 42


def setup_experiment(args):
    utils.set_global_seed(SEED)
    utils.prepare_cudnn(deterministic=True, benchmark=False)

    config = utils.load_config(args.config_path, ordered=True)
    return config


def run_inference(
    config: Dict, *, model: Model, checkpoint: str, loader: torch.utils.data.DataLoader
) -> np.ndarray:
    model.eval()

    runner = CustomRunner()

    embeddings = []

    with tqdm(desc="Running inference", total=len(loader.dataset),) as tq:
        for batch in runner.predict_loader(
            model=model, loader=loader, resume=checkpoint
        ):
            embeddings.append(batch)
            tq.update(batch.size(0))
    return torch.cat(embeddings).cpu().numpy()


def main(args):
    config = setup_experiment(args)
    experiment = Experiment(
        Path(__file__).parent / "input", mode="inference", **config["data_params"],
    )
    inference_loader = experiment.get_loader(args.dataset)

    model = PrototypicalTextClassifier(
        vocab_size=experiment.vocab_size,
        padding_idx=experiment.vocab.pad,
        **config["model_params"],
    )

    if args.mode == "embeddings":
        embeddings = run_inference(
            config, model=model, checkpoint=args.checkpoint, loader=inference_loader
        )
        np.save(args.dataset.replace(".csv", ".embeddings"), embeddings)
    if args.mode == "prototypes":
        targets = torch.cat([x["targets"] for x in inference_loader], dim=0)
        embeddings = run_inference(
            config, model=model, checkpoint=args.checkpoint, loader=inference_loader
        )
        prototypes = model.compute_prototypes(torch.Tensor(embeddings), targets)
        np.save(args.dataset.replace(".csv", ".prototypes"), prototypes.cpu().numpy())


if __name__ == "__main__":
    args = define_args().parse_args()
    main(args)
