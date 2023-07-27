from collections import defaultdict
from typing import List

import torch
from torch.utils.data.dataloader import Dataset, DataLoader
from yacs.config import CfgNode as CN


class _Evaluator:
    pass


class Evaluator(_Evaluator):
    @staticmethod
    def get_default_config():
        C = CN()
        C.name = "Evaluator"
        C.num_workers = 4
        C.batch_size = 16
        C.metrics = []
        return C

    @ staticmethod
    def validate_config(config):
        config = config.clone()
        config.defrost()

        assert type(config.num_workers) == int
        assert config.num_workers >= 0

        assert type(config.batch_size) == int
        assert config.batch_size > 0

        config.freeze()
        return config

    def __init__(self, config, accelerator, datasets: List[Dataset]):
        config = self.validate_config(config)
        self.config = config
        self.accelerator = accelerator
        self.callbacks = defaultdict(list)

        self.dataloaders = {}

        for dataset in datasets:
            self.dataloaders[f"{dataset.category}/{dataset.name}"] = self.accelerator.prepare(
                DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    collate_fn=dataset.collate_fn,
                    shuffle=False
                )
            )

    def add_callback(self, onevent: str, callback, **kwargs):
        self.callbacks[onevent].append(callback)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def trigger_callbacks(self, onevent: str):
        self.event = onevent
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    @torch.no_grad()
    def run(self, trainer):
        self.trigger_callbacks('on_evaluation_start')
        self.steps = trainer.steps
        self.model = trainer.model.eval()
        self.batch_num = 0
        self.total_tasks = trainer.total_tasks

        for name in self.dataloaders:
            for batch in self.dataloaders[name]:
                self.trigger_callbacks('on_batch_start')
                self.batch_losses = {}
                self.batch_logits = {}
                self.batch_labels = {}

                # cache the index of source task.
                task_index = batch[-1][0]
                # create labels
                task_labels = [
                    batch[1] if i == task_index else None
                    for i in range(self.total_tasks)
                ]
                batch[1] = task_labels
                # forward
                task_losses, task_logits = self.model(
                    *batch[:3],
                    single_task=task_index
                )
                # cache several stats
                self.batch_losses[name] = task_losses[task_index].detach().cpu()
                self.batch_logits[name] = task_logits[task_index].detach().cpu()
                self.batch_labels[name] = batch[1][task_index].detach().cpu()

                self.batch_num += 1
                self.batch_loss_info = ",".join(
                    [
                        f"{losses.mean().item()}({name}) "
                        for name, losses in self.batch_losses.items()
                    ]
                )
                self.trigger_callbacks('on_batch_end')

        self.trigger_callbacks('on_evaluation_end')


class VPTEvaluator(_Evaluator):
    @staticmethod
    def get_default_config():
        C = CN()
        C.name = "VPTEvaluator"
        C.num_workers = 4
        C.batch_size = 16
        C.metrics = []
        return C

    def __init__(self, config, accelerator, datasets: List[Dataset]):
        self.config = config
        self.accelerator = accelerator
        self.callbacks = defaultdict(list)

        self.dataloaders = {}

        for dataset in datasets:
            self.dataloaders[f"{dataset.category}/{dataset.name}"] = self.accelerator.prepare(
                DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    collate_fn=dataset.collate_fn
                )
            )

    def add_callback(self, onevent: str, callback, **kwargs):
        self.callbacks[onevent].append(callback)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def trigger_callbacks(self, onevent: str):
        self.event = onevent
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    @torch.no_grad()
    def run(self, trainer):
        self.trigger_callbacks('on_evaluation_start')
        self.steps = trainer.steps
        self.model = trainer.model.eval()
        self.batch_num = 0
        self.total_tasks = trainer.total_tasks

        # while len(dataset_iterators) > 0:
        for name in self.dataloaders:
            for batch in self.dataloaders[name]:
                self.trigger_callbacks('on_batch_start')
                self.batch_losses = {}
                self.batch_logits = {}
                self.batch_labels = {}

                # cache the index of source task.
                task_index = batch[-1][0]
                # create labels
                task_labels = [
                    batch[1] if i == task_index else None
                    for i in range(self.total_tasks)
                ]
                batch[1] = task_labels
                # forward
                task_losses, task_logits = self.model(
                    *batch[:3],
                    single_task=task_index
                )
                # cache several stats
                self.batch_losses[name] = task_losses[task_index].detach()
                self.batch_logits[name] = task_logits[task_index].detach()
                self.batch_labels[name] = batch[1][task_index].detach()

                self.batch_num += 1
                self.batch_loss_info = ",".join(
                    [
                        f"{losses.mean().item()}({name}) "
                        for name, losses in self.batch_losses.items()
                    ]
                )
                self.trigger_callbacks('on_batch_end')
        self.trigger_callbacks('on_evaluation_end')
