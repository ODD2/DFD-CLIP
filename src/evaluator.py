from collections import defaultdict
from typing import List

import torch
from torch.utils.data.dataloader import Dataset, DataLoader
from yacs.config import CfgNode as CN

class Evaluator:
    @staticmethod
    def get_default_config():
        C = CN()
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
            self.dataloaders[dataset.name] = self.accelerator.prepare(
                DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers
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

        dataset_iterators = {
            name:
            iter(dataloader) for name, dataloader in self.dataloaders.items()
        }
    
        while len(dataset_iterators) > 0:
            self.trigger_callbacks('on_batch_start')
            self.batch_losses = {}
            self.batch_logits = {}
            self.batch_labels = {}
            for name in list(dataset_iterators.keys()):
                try:
                    batch = next(dataset_iterators[name])
                except StopIteration:
                    dataset_iterators.pop(name)
                    continue
                
                loss, logits = self.model(*batch)

                # cache several stats
                self.batch_losses[name] = loss.detach()
                self.batch_logits[name] = logits.detach()
                self.batch_labels[name] = batch[1].detach()
            self.batch_num += 1
            self.batch_loss_info = ",".join([f"{losses.mean().item()}({name}_loss) " for name,losses in self.batch_losses.items()])
            self.trigger_callbacks('on_batch_end')
        self.trigger_callbacks('on_evaluation_end')

