import math
import copy
import torch
from collections import defaultdict
from enum import IntEnum, auto

from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, LinearLR
from yacs.config import CfgNode as CN


class TrainMode(IntEnum):
    NORMAL = auto()
    TEACHER = auto()


class LRScheduler(IntEnum):
    ONECYCLE = auto()
    LINEAR = auto()


class _Trainer:
    pass


class Trainer(_Trainer):
    '''
    Generic neural network trainer. Expect to get the optmizer and loss from
    the model instance. (since they're more model-dependent). Metrics and
    loggers are also expected to be added via callbacks.
    '''

    @staticmethod
    def get_default_config():
        C = CN()
        C.name = "Trainer"
        C.max_steps = 100
        C.early_stop = -1
        C.num_workers = 4
        C.batch_size = 16
        C.learning_rate = 1e-3
        C.metrics = []

        # training mode
        C.mode = CN()
        C.mode.type = TrainMode.NORMAL.name.lower()
        # > parameters for ema teacher mode.
        C.mode.teach_at = -1
        C.mode.teach_ema = -1

        # batch per steps
        C.batch_accum = 1

        # learning rate scheduler
        C.lr_scheduler = LRScheduler.ONECYCLE.name.lower()
        return C

    @staticmethod
    def validate_config(config):
        config = config.clone()
        config.defrost()

        assert type(config.max_steps) == int
        assert config.max_steps > 0

        assert type(config.early_stop) == int
        assert config.early_stop == -1 or config.early_stop > 0

        assert type(config.num_workers) == int
        assert config.num_workers >= 0

        assert type(config.batch_size) == int
        assert config.batch_size > 0

        assert type(config.learning_rate) == float
        assert config.learning_rate > 0

        config.mode.type = int(TrainMode[config.mode.type.upper()])
        if config.mode.type == TrainMode.TEACHER:
            assert type(config.mode.teach_at) == int
            assert 0 < config.mode.teach_at
            assert config.mode.teach_at <= config.max_steps

            assert type(config.mode.teach_ema) == float
            assert 0 < config.mode.teach_ema <= 1

        assert type(config.batch_accum) == int
        assert config.batch_accum >= 1

        config.lr_scheduler = int(LRScheduler[config.lr_scheduler.upper()])

        config.freeze()
        return config

    def __init__(self, config, accelerator, model, datasets):
        config = self.validate_config(config)
        self.config = config
        self.accelerator = accelerator
        self.callbacks = defaultdict(list)
        self.model = model

        # prepare learning rate schedulers
        if config.lr_scheduler == LRScheduler.ONECYCLE:
            self.optimizer = model.configure_optimizers(config.learning_rate / 25)
            self.lr_scheduler = OneCycleLR(
                optimizer=self.optimizer,
                max_lr=config.learning_rate,
                total_steps=(config.max_steps * self.accelerator.state.num_processes)
            )

        elif config.lr_scheduler == LRScheduler.LINEAR:
            self.optimizer = model.configure_optimizers(config.learning_rate)
            self.lr_scheduler = LinearLR(
                optimizer=self.optimizer,
                start_factor=0.04,
                total_iters=(config.max_steps/3)
            )

        # pre-cache number of tasks before distributed learning for further usage
        self.total_tasks = len(model.config.out_dim)

        # duplicate model as teacher
        if (self.config.mode.type == TrainMode.TEACHER):
            self.teacher = copy.deepcopy(self.model)
            self.teaching = False

        # let the accelerator prepare the objects for ddp and amp
        if (self.config.mode.type == TrainMode.TEACHER):
            self.model, self.optimizer, self.lr_scheduler, self.teacher = self.accelerator.prepare(
                self.model, self.optimizer, self.lr_scheduler, self.teacher
            )

        else:
            self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.lr_scheduler
            )

        # prepare dataloaders
        self.dataloaders = {}
        for dataset in datasets:
            self.dataloaders[f"{dataset.category}/{dataset.name}"] = self.accelerator.prepare(
                DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    collate_fn=dataset.collate_fn,
                    shuffle=True
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
        self.early_stop_counter = 0

        dataset_iterators = {
            name: iter(self.dataloaders[name]) for name in self.dataloaders
        }

        while True:
            self.trigger_callbacks('on_batch_start')

            # initiliazation
            self.model.zero_grad(set_to_none=True)
            self.model.train()
            self.batch_losses = {name: torch.tensor([]) for name in dataset_iterators.keys()}
            self.batch_logits = {name: torch.tensor([]) for name in dataset_iterators.keys()}
            self.batch_labels = {name: torch.tensor([]) for name in dataset_iterators.keys()}

            # forward pass for batches from each datasets
            for _ in range(self.config.batch_accum):
                with self.accelerator.accumulate(self.model):

                    # iterate over all datasets.
                    for name in dataset_iterators.keys():
                        try:
                            batch = next(dataset_iterators[name])
                        except StopIteration:
                            dataset_iterators[name] = iter(self.dataloaders[name])
                            batch = next(dataset_iterators[name])

                        # cache the index of source task.
                        task_index = batch[-1][0]

                        # prepare labels and flags according to training mode.
                        if (self.config.mode.type == TrainMode.TEACHER and self.teaching):
                            with torch.no_grad():
                                # pseudo labels from EMA teacher
                                _, teacher_logits = self.teacher(
                                    batch[0], [None] * self.total_tasks, batch[2], single_task=-1)

                            # ASSUMPTION: all labels are probabilities.
                            # replace with true labels of the target task
                            task_labels = [
                                batch[1] if i == task_index else teacher_logits[i].softmax(dim=-1)
                                for i in range(self.total_tasks)
                            ]

                            # update the batch data
                            batch[1] = task_labels

                            single_task = None
                        else:
                            task_labels = [
                                batch[1] if i == task_index else None
                                for i in range(self.total_tasks)
                            ]

                            batch[1] = task_labels

                            single_task = task_index

                        # forward and calculate the loss
                        task_losses, task_logits, other_losses = self.model(
                            *batch[:5],
                            train=True,
                            single_task=single_task
                        )

                        # backprop the losses
                        if (self.config.mode.type == TrainMode.TEACHER and self.teaching):
                            self.accelerator.backward(
                                sum([loss.mean() for loss in task_losses]) +
                                sum([other_losses[lname].mean() for lname in other_losses.keys()])
                            )
                        else:
                            self.accelerator.backward(
                                task_losses[task_index].mean() +
                                sum([other_losses[lname].mean() for lname in other_losses.keys()])
                            )

                        # cache output task artifacts
                        self.batch_losses[name] = torch.cat(
                            (self.batch_losses[name], task_losses[task_index].detach().cpu())
                        )
                        self.batch_logits[name] = torch.cat(
                            (self.batch_logits[name], task_logits[task_index].detach().cpu())
                        )
                        self.batch_labels[name] = torch.cat(
                            (self.batch_labels[name], batch[1][task_index].detach().cpu())
                        )

                        # cache auxiliary losses
                        for _k in other_losses.keys():
                            if (not _k in self.batch_losses):
                                self.batch_losses[_k] = torch.tensor([])

                            _losses = None
                            if (len(other_losses[_k].shape) == 1):
                                _losses = other_losses[_k].detach().cpu()
                            elif (len(other_losses[_k].shape) == 2):
                                _losses = other_losses[_k].detach().cpu().mean(1)
                            else:
                                raise NotImplementedError()

                            self.batch_losses[_k] = torch.cat((self.batch_losses[_k], _losses))

                    # update parameter.
                    self.optimizer.step()
                    # update learning rate.
                    self.lr_scheduler.step()
                    # clear gradient after optimizer step
                    self.model.zero_grad(set_to_none=True)
            # accumulate steps
            self.steps += 1

            # update the teacher's parameters under the ema method.
            if (self.config.mode.type == TrainMode.TEACHER):
                for p1, p2 in zip(self.teacher.parameters(), self.model.parameters()):
                    p1.data = (
                        (1 - self.config.mode.teach_ema) * p1.data +
                        self.config.mode.teach_ema * p2.data
                    )

                # activate teaching phase when reaching preset training step
                if (not self.teaching and self.config.mode.teach_at < self.steps):
                    self.teaching = True

            # batch loss infos
            self.batch_loss_info = ",".join(
                [f"{losses.mean().item()}({name}) " for name, losses in self.batch_losses.items()]
            )

            self.trigger_callbacks('on_batch_end')

            if (
                self.steps >= self.config.max_steps or
                (self.config.early_stop > 0 and self.early_stop_counter >= self.config.early_stop)
            ):
                self.trigger_callbacks('on_training_end')
                return


class VPTTrainer(Trainer):
    '''
    Generic neural network trainer. Expect to get the optmizer and loss from
    the model instance. (since they're more model-dependent). Metrics and
    loggers are also expected to be added via callbacks.
    '''

    @staticmethod
    def get_default_config():
        C = Trainer.get_default_config()
        C.name = "VPTTrainer"
        return C

    @staticmethod
    def validate_config(config):
        config = config.clone()
        config = Trainer.validate_config(config)

        assert config.mode.type == TrainMode.NORMAL, "VPT now supports only normal mode training."

        return config

    def __init__(self, config, accelerator, model, datasets):
        config = self.validate_config(config)
        self.config = config
        self.accelerator = accelerator
        self.callbacks = defaultdict(list)

        self.model = model
        if config.lr_scheduler == LRScheduler.ONECYCLE:
            self.optimizer = model.configure_optimizers(config.learning_rate / 25)
            self.lr_scheduler = OneCycleLR(
                optimizer=self.optimizer,
                max_lr=config.learning_rate,
                total_steps=(config.max_steps * self.accelerator.state.num_processes)
            )
        elif config.lr_scheduler == LRScheduler.LINEAR:
            self.optimizer = model.configure_optimizers(config.learning_rate)
            self.lr_scheduler = LinearLR(
                optimizer=self.optimizer,
                start_factor=0.04,
                total_iters=min((config.max_steps / 3), 3000)
            )
        else:
            raise NotImplementedError()

        # pre-cache number of tasks before distributed learning for further usage
        self.total_tasks = len(model.out_dim)

        self.dataloaders = {}

        # let the accelerator prepare the objects for ddp and amp
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )

        for dataset in datasets:
            self.dataloaders[f"{dataset.category}/{dataset.name}"] = self.accelerator.prepare(
                DataLoader(
                    dataset,
                    shuffle=True,
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

    def run(self):
        self.trigger_callbacks('on_training_start')

        self.steps = 0
        self.early_stop_counter = 0

        dataset_iterators = {
            name:
            iter(dataloader) for name, dataloader in self.dataloaders.items()
        }

        while True:
            self.trigger_callbacks('on_batch_start')

            self.model.zero_grad(set_to_none=True)
            self.batch_losses = {}
            self.batch_logits = {}
            self.batch_labels = {}
            self.model.train()
            for name in dataset_iterators.keys():
                try:
                    batch = next(dataset_iterators[name])
                except StopIteration:
                    dataset_iterators[name] = iter(self.dataloaders[name])
                    batch = next(dataset_iterators[name])

                # cache the index of source task.
                task_index = batch[-1][0]

                task_labels = [
                    batch[1] if i == task_index else None
                    for i in range(self.total_tasks)
                ]

                batch[1] = task_labels

                # forward and calculate the loss
                task_losses, task_logits, other_losses = self.model(
                    *batch[:5],
                    train=True,
                    single_task=task_index
                )

                # back propagation
                self.accelerator.backward(
                    task_losses[task_index].mean() +
                    sum([other_losses[lname].mean() for lname in other_losses.keys()])
                )

                # cache output task artifacts
                self.batch_losses[name] = task_losses[task_index].detach()
                self.batch_logits[name] = task_logits[task_index].detach()
                self.batch_labels[name] = batch[1][task_index].detach()

                # cache auxiliary losses
                for _k, _v in other_losses.items():
                    self.batch_losses[_k] = _v.detach().mean()

            self.optimizer.step()
            self.lr_scheduler.step()
            self.model.zero_grad(set_to_none=True)

            self.steps += 1

            # batch loss infos
            self.batch_loss_info = ",".join(
                [f"{losses.mean().item()}({name}) " for name, losses in self.batch_losses.items()]
            )
            self.trigger_callbacks('on_batch_end')

            if (
                self.steps >= self.config.max_steps or
                (self.config.early_stop > 0 and self.early_stop_counter >= self.config.early_stop)
            ):
                self.trigger_callbacks('on_training_end')
                return
