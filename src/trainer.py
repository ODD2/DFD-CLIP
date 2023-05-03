import math
import copy
import torch
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
        C.teach_at = 50
        C.ema_ratio = 0.999
        C.mode="normal"
        return C

    def __init__(self, config, accelerator, model, datasets):
        assert 0 <= config.teach_at <= config.max_steps
        assert config.mode in ["normal","teacher"]
        self.config = config
        self.mode = config.mode
        self.accelerator = accelerator
        self.callbacks = defaultdict(list)

        self.model = model
        self.optimizer = model.configure_optimizers(config.learning_rate / 25)
        # pre-cache number of tasks before distributed learning for further usage
        self.total_tasks = len(model.out_dim)

        self.lr_scheduler = OneCycleLR(
            optimizer=self.optimizer,
            max_lr=config.learning_rate,
            total_steps=(config.max_steps *self.accelerator.state.num_processes)
        )

        self.dataloaders = {}
        self.teaching = False

        if(self.mode == "teacher"):
            self.teacher =  copy.deepcopy(self.model)
        else:
            self.teacher = None
        
        # let the accelerator prepare the objects for ddp and amp
        self.model, self.optimizer, self.lr_scheduler,self.teacher = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler,self.teacher)

        for dataset in datasets:
            self.dataloaders[dataset.name] = self.accelerator.prepare(
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
            self.model.train()
            for name in dataset_iterators.keys():
                try:
                    batch = next(dataset_iterators[name])
                except StopIteration:
                    dataset_iterators[name] = iter(self.dataloaders[name])
                    batch = next(dataset_iterators[name])

                # cache the index of source task.
                task_index = batch[-1][0]
                if(self.teaching):
                    with torch.no_grad():
                        # pseudo labels from EMA teacher
                        _ , teacher_logits = self.teacher(batch[0],[None]*self.total_tasks,batch[2],single_task=-1)

                    # ASSUMPTION: all labels are probabilities.
                    # replace with true labels of the target task
                    task_labels = [ 
                        batch[1] if i == task_index else teacher_logits[i].softmax(dim=-1) 
                        for i in range(self.total_tasks)
                    ]

                    # update the batch data
                    batch[1] = task_labels
                else:
                    task_labels = [ 
                        batch[1] if i == task_index else None
                        for i in range(self.total_tasks)
                    ]

                    batch[1] = task_labels

                
                # forward and calculate the loss
                task_losses, task_logits, recon_loss, match_loss = self.model(
                    *batch[:4],
                    single_task=(task_index if not self.teaching else None),
                    supple=True
                )

                # backprop and update the parameters
                # the loss from the model is should not to be reduced
                # so the duplicate samples can be detected
                if(self.teaching):
                    self.accelerator.backward(sum([loss.mean() for loss in task_losses]))
                else:
                    _comp_inv_loss_pair = torch.cat((recon_loss.unsqueeze(0),match_loss.unsqueeze(0)))
                    # _comp_inv_loss_weight = _comp_inv_loss_pair.softmax(dim=-1)*0.6 + torch.tensor([0.2,0.2],device=_comp_inv_loss_pair.device)
                    _comp_inv_loss_weight = torch.tensor([1.0, 1.0],device=_comp_inv_loss_pair.device)
                    _comp_inv_end_step = 800
                    # _comp_inv_weight = 5*(min(math.log(max(-self.steps + _comp_inv_end_step,1),_comp_inv_end_step),1)*0.8+0.2)
                    _component_balance = 1 - min(self.steps/_comp_inv_end_step,1)*0.99
                    _comp_inv_weight = 100
                    self.accelerator.backward(
                        (
                            (1-_component_balance) * task_losses[task_index].mean() +
                            _component_balance * _comp_inv_weight * (_comp_inv_loss_weight @ _comp_inv_loss_pair)
                        )
                    )

                # cache output artifacts
                self.batch_losses[name] = task_losses[task_index].detach()
                self.batch_logits[name] = task_logits[task_index].detach()
                self.batch_labels[name] = batch[1][task_index].detach()
                self.batch_losses["recon"] = recon_loss.detach()
                self.batch_losses["match"] = match_loss.detach()
            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.model.zero_grad()

            if(self.mode == "teacher"):
                # update the teacher's parameters in the EMA manner.
                for p1,p2 in zip(self.teacher.parameters(),self.model.parameters()):
                    p1.data = (1-self.config.ema_ratio)*p1.data+self.config.ema_ratio*p2.data

            self.steps += 1

            # activate teaching process 
            if(self.mode == "teacher" and not self.teaching and self.config.teach_at < self.steps):
                self.teaching = True
            
            # batch loss infos
            self.batch_loss_info = ",".join([f"{losses.mean().item()}({name}_loss) " for name,losses in self.batch_losses.items()])
            self.trigger_callbacks('on_batch_end')

            if self.steps >= self.config.max_steps:
                self.trigger_callbacks('on_training_end')
                return

            # self.trigger_callbacks('on_epoch_end')

