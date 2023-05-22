import wandb
import torch
import evaluate


class rmse:
    def __init__(self):
        self.expects = []
        self.labels = []

    def add_batch(self, pred_labels, pred_probs, labels):
        b, w = pred_probs.shape
        self.expects.append(pred_probs @ torch.tensor([i for i in range(w)]).float().to(pred_probs.device))
        self.labels.append(labels)

    def compute(self):
        if (not len(self.expects) == len(self.labels) or len(self.expects) == 0):
            raise Exception("nothing to compute, please atleast call add_batch once before compute")
        else:
            expects = torch.cat(self.expects)
            labels = torch.cat(self.labels)
            self.expects = []
            self.labels = []
            return {
                f"{self.__class__.__name__}":
                torch.sqrt(torch.sum(torch.pow(expects - labels, 2) / len(expects))).item()
            }


class mse:
    def __init__(self):
        self.calc = evaluate.load("mse", "multilist")

    def add_batch(self, pred_labels, pred_probs, labels):
        self.calc.add_batch(
            references=labels.to(torch.float32),
            predictions=pred_probs.to(torch.float32)
        )

    def compute(self):
        return self.calc.compute()


class roc_auc:
    def __init__(self):
        self.calc = evaluate.load("roc_auc")

    def add_batch(self, pred_labels, pred_probs, labels):
        self.calc.add_batch(
            references=labels.to(torch.float32),
            prediction_scores=pred_probs[:, 1].to(torch.float32)
        )

    def compute(self):
        return self.calc.compute()


class accuracy:
    def __init__(self):
        self.calc = evaluate.load("accuracy")

    def add_batch(self, pred_labels, pred_probs, labels):
        self.calc.add_batch(
            references=labels.to(torch.float32),
            predictions=pred_labels.to(torch.float32)
        )

    def compute(self):
        return self.calc.compute()


@torch.no_grad()
def init_metrics(agent):
    agent.calcs = {
        cfg.name: {
            setup: globals()[setup]()
            for setup in cfg.types
        }
        for cfg in agent.config.metrics
    }
    # agent.roc_auc_calc = evaluate.load("roc_auc")
    agent.losses = {
    }


@torch.no_grad()
def update_metrics(agent):

    pred_labels = {
        name: logits.argmax(dim=-1)
        for name, logits in agent.batch_logits.items()
    }
    pred_probs = {
        name: logits.softmax(dim=-1)
        for name, logits in agent.batch_logits.items()
    }

    pred_labels, pred_probs, batch_labels, batch_losses = agent.accelerator.gather_for_metrics((
        pred_labels, pred_probs, agent.batch_labels, agent.batch_losses))

    if not agent.accelerator.is_local_main_process:
        return

    # update metrics
    for name, labels in batch_labels.items():
        for metric in agent.calcs[name].values():
            metric.add_batch(
                pred_labels=pred_labels[name].to(torch.float32),
                pred_probs=pred_probs[name].to(torch.float32),
                labels=labels.to(torch.float32)
            )
    # update losses
    for name, loss in batch_losses.items():
        if not name in agent.losses:
            agent.losses[name] = []
        agent.losses[name].append(loss.mean().item())


@torch.no_grad()
def compute_metrics(agent):
    if agent.steps % agent.training_eval_interval:
        return
    agent.compute_losses = {}
    agent.compute_metrics = {}

    # compute metrics
    for lname in agent.calcs.keys():
        for mname, metric in agent.calcs[lname].items():
            agent.compute_metrics[f"metric/{lname}/{mname}"] = metric.compute()[mname]

    # compute losses
    for lname in agent.losses.keys():
        agent.compute_losses[f"loss/{lname}"] = sum(agent.losses[lname]) / len(agent.losses[lname])
        agent.losses[lname].clear()

    agent.accelerator.print(
        {
            **agent.compute_losses,
            **agent.compute_metrics
        }
    )

    wandb.log(
        {
            **{
                f'{type(agent).__name__}/{lname}'.lower(): value
                for lname, value in agent.compute_losses.items()
            },
            **{
                f'{type(agent).__name__}/{mname}'.lower(): value
                for mname, value in agent.compute_metrics.items()
            },
        },
        step=agent.steps
    )
