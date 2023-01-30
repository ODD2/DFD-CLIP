from collections import defaultdict

from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from yacs.config import CfgNode as CN

class Trainer:
    '''
    Generic neural network trainer. Expect to get the optmizer and loss from
    the model instance. (since they're more model-dependent). Metrics and
    loggers are also expected to be added via callbacks.
    '''

    @staticmethod
    def get_default_config():
        C = CN()
        C.max_steps = 100
        C.num_workers = 4
        C.batch_size = 16
        C.learning_rate = 1e-3
        return C

    def __init__(self, config, accelerator, model, dataset):
        self.config = config
        self.accelerator = accelerator
        self.callbacks = defaultdict(list)

        self.model = model
        self.optimizer = model.configure_optimizers(config.learning_rate / 25)
        self.lr_scheduler = OneCycleLR(optimizer=self.optimizer,
                                       max_lr=config.learning_rate,
                                       total_steps=(config.max_steps *
                                                    self.accelerator.state.num_processes))
        self.dataloader = DataLoader(dataset,
                                     shuffle=True,
                                     batch_size=config.batch_size,
                                     num_workers=config.num_workers)

        # let the accelerator prepare the objects for ddp and amp
        self.model, self.optimizer, self.dataloader, self.lr_scheduler = self.accelerator.prepare(
        self.model, self.optimizer, self.dataloader, self.lr_scheduler)

    def add_callback(self, onevent: str, callback, **kwargs):
        self.callbacks[onevent].append(callback)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def trigger_callbacks(self, onevent: str):
        self.event = onevent
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        self.trigger_callbacks('on_training_start')

        self.model.train()
        self.steps = 0

        while True:
            self.trigger_callbacks('on_epoch_start')
            for batch in self.dataloader:
                self.trigger_callbacks('on_batch_start')

                # forward and calculate the loss
                loss, logits = self.model(*batch)

                # backprop and update the parameters
                # the loss from the model is should not to be reduced
                # so the duplicate samples can be detected
                self.model.zero_grad()
                self.accelerator.backward(loss.mean())
                self.optimizer.step()
                self.lr_scheduler.step()
 
                # cache output artifacts
                self.loss = loss.detach()
                self.logits = logits.detach()
                self.labels = batch[1]

                self.steps += 1
                self.trigger_callbacks('on_batch_end')

                if self.steps >= self.config.max_steps:
                    self.trigger_callbacks('on_training_end')
                    return

            self.trigger_callbacks('on_epoch_end')

