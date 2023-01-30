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
        return C

    def __init__(self, config, accelerator, datasets: List[Dataset]):
        self.config = config
        self.accelerator = accelerator
        self.callbacks = defaultdict(list)

        self.dataloaders = [
            DataLoader(dataset,
                       batch_size=config.batch_size,
                       num_workers=config.num_workers)
        for dataset in datasets]

        # let the accelerator prepare the dataloader
        # for ddp and amp if len(datalaoder) > 0
        self.dataloaders = self.dataloaders and self.accelerator.prepare(*self.dataloaders)
        # in the case where there is only one evaluation dataset
        self.dataloaders = self.dataloaders if isinstance(self.dataloaders, (tuple, list)) else [self.dataloaders]

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

        for self.dataloader in self.dataloaders:
            self.trigger_callbacks('on_dataloader_start')

            for batch in self.dataloader:
                self.trigger_callbacks('on_batch_start')

                loss, logits = self.model(*batch)

                # cache several stats
                self.loss = loss
                self.logits = logits
                self.labels = batch[1]

                self.trigger_callbacks('on_batch_end')

            self.trigger_callbacks('on_dataloader_end')

        self.trigger_callbacks('on_evaluation_end')

        # XXX: if we don't set the gradient_state ourselves
        #      the state remains at the end of evaluation
        #      which incorrectly drops samples when `gather_for_metrics`
        #      in the following training
        #      see https://github.com/huggingface/accelerate/issues/960
        self.accelerator.gradient_state.end_of_dataloader = False

