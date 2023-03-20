import torch
import  evaluate

@torch.no_grad()
def init_metrics(agent):
    agent.mse_calc = evaluate.load("mse","multilist")
    # agent.roc_auc_calc = evaluate.load("roc_auc")
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

    agent.mse_calc.add_batch(
        references=labels.to(torch.float32),
        predictions=pred_probs.to(torch.float32)
    )
    agent.losses.append(losses.mean().item())


@torch.no_grad()
def compute_metrics(agent):
    if agent.steps % agent.training_eval_interval:
        return

    agent.mse = agent.mse_calc.compute()['mse']
    agent.loss_avg = sum(agent.losses) / len(agent.losses)

    agent.accelerator.print({'mse': agent.mse,'loss_avg':agent.loss_avg})

    agent.losses = []

