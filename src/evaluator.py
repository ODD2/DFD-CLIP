import logging
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
                    batch_size=config.batch_size if not dataset.pack else 1,
                    num_workers=config.num_workers,
                    collate_fn=dataset.collate_fn if not dataset.pack else (lambda x: x[0]),
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

                # forward
                if self.dataloaders[name].dataset.pack:
                    N = self.config.batch_size
                    clips, label, masks, meta, task_index = *batch[:3], batch[-2], batch[-1]
                    if (len(clips) == 0):
                        logging.error(f"video '{i}' cannot provide clip for inference, skipping...")
                        _losses = torch.tesor([0.7])
                        _logits = torch.tensor([[0.5, 0.5]])
                        _labels = torch.tensor([label], device="cpu")
                    else:
                        # cache the index of source task.
                        task_index = batch[-1]
                        video_logits = []
                        video_losses = []
                        for i in range(0, len(clips), N):
                            task_losses, task_logits = self.model(
                                torch.stack(clips[i:i + N]).to(self.accelerator.device),
                                [
                                    (torch.tensor([label]*len(clips[i:i + N])).to(self.accelerator.device))
                                    if i == task_index else None
                                    for i in range(self.total_tasks)
                                ],
                                torch.stack(masks[i:i + N]).to(self.accelerator.device)
                            )

                            video_losses.append(task_losses[task_index].detach().to("cpu"))
                            video_logits.append(task_logits[task_index].detach().to("cpu"))

                        _logits = torch.cat(video_logits).softmax(dim=-1).mean(dim=0).unsqueeze(0)
                        _losses = torch.cat(video_losses).mean(0).unsqueeze(0)
                        _labels = torch.tensor([label], device="cpu")
                else:
                    # cache the index of source task.
                    task_index = batch[-1][0]
                    # create labels
                    task_labels = [
                        batch[1] if i == task_index else None
                        for i in range(self.total_tasks)
                    ]
                    batch[1] = task_labels

                    task_losses, task_logits = self.model(
                        *batch[:3],
                        single_task=task_index
                    )
                    _losses = task_losses[task_index].detach().cpu()
                    _logits = task_logits[task_index].detach().cpu()
                    _labels = batch[1][task_index].detach().cpu()
                # cache several stats
                self.batch_losses[name] = _losses
                self.batch_logits[name] = _logits
                self.batch_labels[name] = _labels

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
