import builtins

import torch
from ..trainer import Trainer

@torch.no_grad()
def update_trackers(agent):
    if agent.steps % agent.training_eval_interval:
        return

    agent.accelerator.log(
        agent.compute_losses,
        step=agent.steps
    )

    # Comment out the following lines due to the new training structure utilize more than one dataset.

    # if not isinstance(agent, Trainer):
    #     return

    # agent.accelerator.log(
    #     {
    #         # f'epoch/{agent.dataloader.dataset.name}': agent.steps * agent.dataloader.total_batch_size / agent.dataloader.total_dataset_length,
    #         f'lr/{agent.dataloader.dataset.name}': agent.optimizer.param_groups[0]['lr'],
    #     },
    #     step=agent.steps,
    # )


# model saver
@torch.no_grad()
def add_main_metric(agent):
    agent.current_main_metrics.append(getattr(agent, agent.main_metric))

@torch.no_grad()
def cache_best_model(agent):
    target_metrics = [value for name,value in agent.compute_metrics.items() if  agent.main_metric in name]
    main_metric =  sum(target_metrics) / max(len(target_metrics) ,0)
    current_best = getattr(agent, 'best_main_metric', main_metric)

    if getattr(builtins, agent.compare_fn)(main_metric, current_best) == main_metric:
        # update best
        agent.accelerator.print(f'best model updated with {agent.main_metric} of', main_metric,
                                f'(past SOTA: {current_best})')
        agent.best_main_metric = main_metric
        agent.best_model_state = agent.accelerator.unwrap_model(agent.model).state_dict()
        agent.best_model_state = {k: v.cpu() for k, v in agent.best_model_state.items()}

