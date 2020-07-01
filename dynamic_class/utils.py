import os
import logging
from pathlib import Path
import shutil

from catalyst.dl.utils import (
    get_utcnow_time,
    get_short_hash,
    set_global_seed,
    prepare_cudnn,
    load_config,
    dump_environment,
    load_checkpoint,
    unpack_checkpoint,
    get_device,
    any2device,
)


def _tricky_dir_copy(dir_from, dir_to, ignored_dirs):
    os.makedirs(dir_to, exist_ok=True)
    shutil.rmtree(dir_to)
    shutil.copytree(dir_from, dir_to, ignore=shutil.ignore_patterns(*ignored_dirs))


def dump_code(expdir: str, logdir: str, ignored_dirs=["logs", "input"]):
    assert isinstance(expdir, str)
    assert isinstance(logdir, str)
    expdir = expdir[:-1] if expdir.endswith("/") else expdir
    new_src_dir = f"code"

    old_expdir = os.path.abspath(expdir)
    expdir_ = os.path.basename(old_expdir)
    new_expdir = os.path.join(logdir, new_src_dir, expdir_)

    _tricky_dir_copy(old_expdir, new_expdir, ignored_dirs)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    return logger
