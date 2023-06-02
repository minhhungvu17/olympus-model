from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import numpy as np

import signal
import random


__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../..")))


import copy
from torch.utils.data import Dataset, DataLoader, BatchSampler, DistributedSampler

from pytorchocr.data.imaug import transform, create_operators
from pytorchocr.data.simple_dataset import SimpleDataSet

# from pytorchocr.data.lmdb_dataset import LMDBDateSet

__all__ = ["build_dataloader", "transform", "create_operators"]


def tem_mp(sig_num, frame):
    """
    Kill all child processes
    Args:
        sig_num (_type_): _description_
        frame (_type_): _description_
    """
    pid = os.getpid()
    pgid = os.getpgid(os.getpid())
    print("main proc {} exit, kill process group " "{}".format(pid, pgid))
    os.killpg(pgid, signal.SIGKILL)


def build_dataloader(config, mode, logger, seed=None):
    config = copy.deepcopy(config)

    support_dict = ["SimpleDataSet"]
    module_name = config[mode]["dataset"]["name"]
    assert module_name in support_dict, Exception(
        "Dataset only support {}".format(support_dict)
    )
    assert mode in ["Train", "Val", "Test"], "Mode should be Train, Val or Test."

    dataset = eval(module_name)(config, mode, logger, seed)

    loader_config = config[mode]["loader"]
    batch_size = loader_config["batch_size_per_card"]
    drop_last = loader_config["drop_last"]
    shuffle = loader_config["shuffle"]
    num_workers = loader_config["num_workers"]

    # if mode == "Train":
    #     batch_sampler = DistributedSampler(
    #         dataset=dataset, shuffle=shuffle, drop_last=drop_last
    #     )
    # else:
    #     batch_sampler = BatchSampler(
    #         dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    #     )

    if "collate_fn" in loader_config:
        from . import collate_fn

        collate_fn = getattr(collate_fn, loader_config["collate_fn"])
    else:
        collate_fn = None

    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=None,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    signal.signal(signal.SIGINT, tem_mp)
    signal.signal(signal.SIGTERM, tem_mp)

    return data_loader
