import torch
import  evaluate

@torch.no_grad()
def init_metrics(agent):
    agent.accuracy_calc = evaluate.load("accuracy")
    agent.roc_auc_calc = evaluate.load("roc_auc")
    agent.losses = []


@torch.no_grad()
def update_metrics(agent):
    pred_labels = agent.logits.argmax(dim=-1)
    pred_probs = agent.logits.softmax(dim=-1)

    pred_labels, pred_probs, labels, losses = agent.accelerator.gather_for_metrics((
    pred_labels, pred_probs, agent.labels, agent.loss))

    if not agent.accelerator.is_local_main_process:
        return

    if losses.numel() == 0:
        # XXX: this seems to be a bug of accelerate, the entire last batch is
        #      consided duplicate by gather_for_metrics. This happens only when
        #      dataset gets perfectly sharded.
        #      see https://github.com/huggingface/accelerate/issues/952
        #
        # Update: fixed in https://github.com/huggingface/accelerate/pull/982
        #         waiting for the next accelerate release
        return

    agent.accuracy_calc.add_batch(references=labels, predictions=pred_labels)
    agent.roc_auc_calc.add_batch(references=labels, prediction_scores=pred_probs[:, 1]) # prob of real class
    agent.losses.append(losses.mean().item())


@torch.no_grad()
def compute_metrics(agent):
    if agent.steps % agent.training_eval_interval:
        return

    agent.accuracy = agent.accuracy_calc.compute()['accuracy']
    agent.roc_auc = agent.roc_auc_calc.compute()['roc_auc']
    agent.loss_avg = sum(agent.losses) / len(agent.losses)

    agent.accelerator.print({'accuracy': agent.accuracy, 'roc_auc': agent.roc_auc, 'loss': agent.loss_avg})

    agent.losses = []

