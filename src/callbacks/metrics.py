import torch
import  evaluate
import wandb

class mse:
    def __init__(self):
        self.calc = evaluate.load("mse","multilist")

    def add_batch(self,pred_labels,pred_probs,labels):
        self.calc.add_batch(
                references=labels.to(torch.float32),
                predictions=pred_probs.to(torch.float32)
        )

    def compute(self):
        return self.calc.compute()
    
class roc_auc:
    def __init__(self):
        self.calc = evaluate.load("roc_auc")

    def add_batch(self,pred_labels,pred_probs,labels):
        self.calc.add_batch(
                references=labels.to(torch.float32),
                prediction_scores=pred_probs[:,1].to(torch.float32)
        )

    def compute(self):
        return self.calc.compute()
    
class accuracy:
    def __init__(self):
        self.calc = evaluate.load("accuracy")

    def add_batch(self,pred_labels,pred_probs,labels):
        self.calc.add_batch(
                references=labels.to(torch.float32),
                predictions=pred_labels.to(torch.float32)
        )

    def compute(self):
        return self.calc.compute()
    
@torch.no_grad()
def init_metrics(agent):
    agent.calcs ={
        cfg.name: {
            setup: globals()[setup]()            
            for setup in cfg.types
        }
        for cfg in agent.config.metrics  
    }
    # agent.roc_auc_calc = evaluate.load("roc_auc")
    agent.losses = {
        name: []
        for name in agent.calcs.keys()
    }


@torch.no_grad()
def update_metrics(agent):

    pred_labels = {
        name: logits.argmax(dim=-1)
        for name,logits in agent.batch_logits.items()
    }
    pred_probs = {
        name: logits.softmax(dim=-1)
        for name,logits in agent.batch_logits.items()
    }

    pred_labels, pred_probs, batch_labels, batch_losses = agent.accelerator.gather_for_metrics((
    pred_labels, pred_probs, agent.batch_labels, agent.batch_losses))

    if not agent.accelerator.is_local_main_process:
        return
    
    for name,labels in batch_labels.items():
        for metric in agent.calcs[name].values():
            metric.add_batch(
                pred_labels=pred_labels[name].to(torch.float32),
                pred_probs=pred_probs[name].to(torch.float32),
                labels=labels.to(torch.float32)
            )
    for name,loss in batch_losses.items():
        agent.losses[name].append(loss.mean().item())


@torch.no_grad()
def compute_metrics(agent):
    if agent.steps % agent.training_eval_interval:
        return
    agent.compute_losses = {}
    agent.compute_metrics = {}
    for lname in agent.losses.keys():
        for mname,metric  in agent.calcs[lname].items():
            agent.compute_metrics[f"metric/{lname}/{mname}"] = metric.compute()[mname]
        
        agent.compute_losses[f"loss/{lname}"] = sum(agent.losses[lname]) / len(agent.losses[lname])
        agent.losses[lname].clear()

    agent.accelerator.print(
        {
            **agent.compute_losses,
            **agent.compute_metrics
        }
    )
    
    wandb.log(
        data={
            **{
                f'{type(agent).__name__}/{lname}'.lower():value
                for lname,value in agent.compute_losses.items()
            },
            **{
                f'{type(agent).__name__}/{mname}'.lower(): value
                for mname, value in agent.compute_metrics.items()
            },
        },
        step=agent.steps
    )
        



