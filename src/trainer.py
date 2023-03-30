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
        C.metrics = []
        return C

    def __init__(self, config, accelerator, model, datasets):
        self.config = config
        self.accelerator = accelerator
        self.callbacks = defaultdict(list)

        self.model = model
        self.optimizer = model.configure_optimizers(config.learning_rate / 25)

        self.lr_scheduler = OneCycleLR(
            optimizer=self.optimizer,
            max_lr=config.learning_rate,
            total_steps=(config.max_steps *self.accelerator.state.num_processes)
        )

        self.dataloaders = {}
      
        # let the accelerator prepare the objects for ddp and amp
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler)
        
        for dataset in datasets:
            self.dataloaders[dataset.name] = self.accelerator.prepare(
                DataLoader(
                    dataset,
                    shuffle=True,
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

    def run(self):
        self.trigger_callbacks('on_training_start')

        self.steps = 0
        dataset_iterators = {
            name:
            iter(dataloader) for name, dataloader in self.dataloaders.items()
        }
        while True:
            # self.trigger_callbacks('on_epoch_start')
            self.trigger_callbacks('on_batch_start')
            ############################
            self.model.zero_grad()
            self.batch_losses = {}
            self.batch_logits = {}
            self.batch_labels = {}
            for name in dataset_iterators.keys():
                try:
                    batch = next(dataset_iterators[name])
                except StopIteration:
                    dataset_iterators[name] = iter(self.dataloaders[name])
                    batch = next(dataset_iterators[name])

                self.model.train()
            

                # forward and calculate the loss
                loss, logits = self.model(*batch)

                # backprop and update the parameters
                # the loss from the model is should not to be reduced
                # so the duplicate samples can be detected
                self.accelerator.backward(loss.mean())
                
                # cache output artifacts
                self.batch_losses[name] = loss.detach()
                self.batch_logits[name] = logits.detach()
                self.batch_labels[name] = batch[1].detach()
            
            self.optimizer.step()
            self.lr_scheduler.step()


            self.steps += 1
            self.batch_loss_info = ",".join([f"{losses.mean().item()}({name}_loss) " for name,losses in self.batch_losses.items()])
            self.trigger_callbacks('on_batch_end')

            if self.steps >= self.config.max_steps:
                self.trigger_callbacks('on_training_end')
                return

            # self.trigger_callbacks('on_epoch_end')

