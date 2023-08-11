import wandb
import builtins
import re
import torch
import logging
from ..trainer import _Trainer


@torch.no_grad()
def update_trackers(agent):
    if agent.steps % agent.training_eval_interval:
        return

    if not isinstance(agent, _Trainer):
        return

    wandb.log(
        {
            f'lr': agent.optimizer.param_groups[0]['lr'],
        },
        step=agent.steps,
    )


@torch.no_grad()
def cache_best_model(agent):
    target_metrics = [value for name, value in agent.phase_metrics.items() if re.search(agent.main_metric, name)]
    if (len(target_metrics) > 0):
        main_metric = sum(target_metrics) / max(len(target_metrics), 1)
        current_best = getattr(agent, 'best_main_metric', main_metric)

        if (
            getattr(builtins, agent.compare_fn)(main_metric, current_best) == main_metric or
            abs(main_metric - current_best) < 1e-3
        ):
            # update best
            logging.info(f'best model updated with "{agent.main_metric}" of {main_metric}(past SOTA: {current_best})')
            agent.best_main_metric = main_metric
            agent.best_model_state = agent.accelerator.unwrap_model(agent.model).state_dict()
            agent.best_model_state = {k: v.cpu() for k, v in agent.best_model_state.items()}

    # update latest
    agent.last_model_state = agent.accelerator.unwrap_model(agent.model).state_dict()
    agent.last_model_state = {k: v.cpu() for k, v in agent.last_model_state.items()}
