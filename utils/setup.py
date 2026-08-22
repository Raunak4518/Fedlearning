"""
utils/setup.py

`setup_experiment(args)` is the one call every top-level script makes
before training starts. It resolves --dataset, --target_models, and
--gen_model through their registries, applies the long-tail + Dirichlet
partition, and returns everything the federated loop needs -- nothing in
this file hardcodes a dataset name, an architecture name, or a class
count; every one of those comes from `args` and the registries in
datasets/, targetNetModels/, and generators/.
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from datasets.get_dataset import get_dataset
from generators.base import GEN_REGISTRY
import generators.ccvae  # noqa: F401 -- registers "vae"
import generators.ccgan  # noqa: F401 -- registers "gan"
import generators.cddpm  # noqa: F401 -- registers "ddpm"
from sampling.partition import (class_counts_long_tailed, client_class_counts, frequency_bucket_labels,
                                 partition_dataset)
from targetNetModels.nets import NET_REGISTRY


@dataclass
class Experiment:
    dataset_train: object
    dataset_test: object
    meta: object
    train_labels: np.ndarray
    dict_users: Dict[int, List[int]]
    dev_spec_idx: List[int]                 # dev_spec_idx[client_id] -> which target-net architecture group
    target_net_classes: List[type]          # one class per architecture group (length == args.num_models)
    ws_glob: List[dict]                     # initial global weights, one per architecture group
    class_counts: Dict[int, int]            # global per-class counts AFTER the long-tail schedule
    buckets: Dict[int, str]                 # class_id -> 'head'/'medium'/'tail'
    client_class_counts: Dict[int, Dict[int, int]]  # per-client per-class counts, for Mechanism A
    gen_cls: type = None


def setup_experiment(args) -> Experiment:
    dataset_train, dataset_test, meta, train_labels = get_dataset(args)
    num_classes, in_channels, img_size = meta.num_classes, meta.in_channels, meta.native_img_size

    dict_users = partition_dataset(train_labels, num_classes, args)
    class_counts = class_counts_long_tailed(train_labels, num_classes, args.imbalance_factor, args.max_per_class)
    buckets = frequency_bucket_labels(class_counts)
    per_client_counts = {cid: client_class_counts(train_labels, idxs, num_classes)
                          for cid, idxs in dict_users.items()}

    dev_spec_idx = [i % args.num_models for i in range(args.num_users)]
    target_net_classes = [NET_REGISTRY.get(name) for name in args.target_models_list]

    torch.manual_seed(args.seed)
    ws_glob = []
    for cls in target_net_classes:
        net = cls(in_channels=in_channels, num_classes=num_classes, img_size=img_size).to(args.device)
        ws_glob.append(net.state_dict())

    gen_cls = GEN_REGISTRY.get(args.gen_model) if args.aid_by_gen else None

    return Experiment(
        dataset_train=dataset_train, dataset_test=dataset_test, meta=meta, train_labels=train_labels,
        dict_users=dict_users, dev_spec_idx=dev_spec_idx, target_net_classes=target_net_classes,
        ws_glob=ws_glob, class_counts=class_counts, buckets=buckets, client_class_counts=per_client_counts,
        gen_cls=gen_cls,
    )


def build_target_net(exp: Experiment, client_id: int, args) -> torch.nn.Module:
    group = exp.dev_spec_idx[client_id]
    cls = exp.target_net_classes[group]
    net = cls(in_channels=exp.meta.in_channels, num_classes=exp.meta.num_classes,
              img_size=exp.meta.native_img_size).to(args.device)
    net.load_state_dict(exp.ws_glob[group])
    return net


def build_generator(exp: Experiment, args) -> torch.nn.Module:
    return exp.gen_cls(num_classes=exp.meta.num_classes, in_channels=exp.meta.in_channels,
                        img_size=exp.meta.native_img_size, args=args).to(args.device)
