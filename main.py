import os
import random
import re
import argparse
from datetime import timedelta

import numpy as np
import torch
from accelerate import Accelerator
from yacs.config import CfgNode as CN

from src.models import Detector
from src.datasets import FFPP
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.callbacks.timer import start_timer, end_timer
from src.callbacks.metrics import init_metrics, update_metrics, compute_metrics
from src.callbacks.tracking import update_trackers, add_main_metric, cache_best_model

def get_config(config_file):
    C = CN()

    # system
    C.system = CN()
    C.system.mixed_precision = 'no' # no | fp16 | bf16
    C.system.seed = 0
    C.system.deterministic_training = False
    C.system.training_eval_interval = 10
    C.system.evaluation_interval = 10

    # tracking
    C.tracking = CN()
    C.tracking.enabled = False
    C.tracking.directory = 'test-logs'
    C.tracking.project_name = None
    C.tracking.default_project_prefix = 'version'
    C.tracking.tool = 'all'
    C.tracking.main_metric = 'roc_auc' # accuracy | roc_auc
    C.tracking.compare_fn = 'max' # max | min

    # data
    C.data = CN()
    C.data.train = FFPP.get_default_config()
    C.data.eval = []

    # model
    C.model = Detector.get_default_config()

    # trainer
    C.trainer = Trainer.get_default_config()

    # evaluator
    C.evaluator = Evaluator.get_default_config()

    # load additional configs from file
    if config_file is not None:
        C.merge_from_file(config_file)

    # post-process eval datasets
    C.data.eval = [FFPP.get_default_config().merge_from_other_cfg(d_eval) for d_eval in C.data.eval]

    C.freeze()

    # some sanity check
    for d_eval in C.data.eval:
        assert 'name' in d_eval

    return C


def register_trainer_callbacks(config, trainer, **kwargs):
    def evaluation_proxy(trainer):
        if trainer.steps % trainer.evaluation_interval:
            return
        kwargs['evaluator'].run(trainer)

    def save_model(trainer):
        trainer.accelerator.save(kwargs['evaluator'].best_model_state,
                                 os.path.join(trainer.accelerator.tracking_dir, 'best_weights.pt'))

    # timer
    timer_events = ['training', 'epoch', 'batch']
    trainer.add_callback('on_training_start', lambda _: None, timer={evt: 0 for evt in timer_events})
    for event in timer_events:
        trainer.add_callback(f'on_{event}_start', start_timer)
        trainer.add_callback(f'on_{event}_end', end_timer)

    # metrics
    trainer.add_callback('on_batch_end', update_metrics)
    if trainer.accelerator.is_local_main_process:
        trainer.add_callback('on_training_start', init_metrics)
        trainer.add_callback('on_batch_end', compute_metrics, training_eval_interval=config.system.training_eval_interval)

    # tracker
    if config.tracking.enabled and trainer.accelerator.is_local_main_process:
        trainer.add_callback('on_batch_end', update_trackers)
        trainer.add_callback('on_training_end', save_model)

    # evaluator
    trainer.add_callback('on_batch_end', evaluation_proxy, evaluation_interval=config.system.evaluation_interval)

    # stdout logger
    trainer.add_callback('on_batch_end',
        lambda trainer: trainer.accelerator.print(f'{trainer.steps} | loss {trainer.loss.mean().item()}, {trainer.batch_duration:.2f}s'))
    trainer.add_callback('on_epoch_end',
        lambda trainer: trainer.accelerator.print(f'epoch takes {trainer.epoch_duration:.2f}s'))
    trainer.add_callback('on_training_end',
        lambda trainer: trainer.accelerator.print(f'training completed in {timedelta(seconds=trainer.training_duration)}'))


def register_evaluator_callbacks(config, evaluator, **kwargs):
    def clear_current_main_metrics(evaluator):
        evaluator.current_main_metrics = []

    # timer
    timer_events = ['evaluation', 'dataloader']
    evaluator.add_callback('on_evaluation_start', lambda _: None, timer={evt: 0 for evt in timer_events})
    evaluator.add_callback('on_evaluation_start', lambda evaluator: evaluator.accelerator.print('evaluation start'))
    for event in timer_events:
        evaluator.add_callback(f'on_{event}_start', start_timer)
        evaluator.add_callback(f'on_{event}_end', end_timer)

    # metrics
    evaluator.add_callback('on_batch_end', update_metrics)
    if evaluator.accelerator.is_local_main_process:
        evaluator.add_callback('on_evaluation_start', init_metrics)
        evaluator.add_callback('on_dataloader_end', compute_metrics, training_eval_interval=1)

    # tracker
    if config.tracking.enabled and evaluator.accelerator.is_local_main_process:
        evaluator.add_callback('on_dataloader_end', update_trackers)

        # model saver
        evaluator.add_callback('on_evaluation_start', clear_current_main_metrics,
                                                      main_metric=config.tracking.main_metric,
                                                      compare_fn=config.tracking.compare_fn,
                                                      current_main_metrics=[])
        evaluator.add_callback('on_dataloader_end', add_main_metric)
        evaluator.add_callback('on_evaluation_end', cache_best_model)

    # stdout logger
    evaluator.add_callback('on_dataloader_end',
        lambda evaluator: evaluator.accelerator.print(f'evaluation of {evaluator.dataloader.dataset.name} completed in {evaluator.dataloader_duration:.2f}s'))
    evaluator.add_callback('on_evaluation_end',
        lambda evaluator: evaluator.accelerator.print(f'evaluation completed in {evaluator.evaluation_duration:.2f}'))


def main(config_file):
    config = get_config(config_file)

    # initialize accelerator and trackers (if enabled)
    accelerator = init_accelerator(config)
    accelerator.print(config.dump())

    # set random seed for deterministic training
    if config.system.deterministic_training:
        set_seed(config.system.seed)

    # initialize model
    model = Detector(config.model, config.data.train.num_frames, accelerator)

    # initialize datasets
    train_dataset = FFPP(config.data.train, model.transform, accelerator, 'train')
    accelerator.print(f'Dataset {train_dataset.name} initialized with {len(train_dataset)} samples\n')

    eval_datasets = [FFPP(cfg, model.transform, accelerator, 'val') for cfg in config.data.eval]
    for dataset in eval_datasets:
        accelerator.print(f'Dataset {dataset.name} initialized with {len(dataset)} samples\n')

    # initialize trainer and evaluator
    trainer = Trainer(config.trainer, accelerator, model, train_dataset)
    evaluator = Evaluator(config.evaluator, accelerator, eval_datasets)

    # register callbacks
    register_trainer_callbacks(config, trainer, evaluator=evaluator)
    register_evaluator_callbacks(config, evaluator)

    # start training
    trainer.run()

    # finished
    if config.tracking.enabled:
        accelerator.end_training()


def init_accelerator(config):
    # init huggingface accelerator
    if not config.tracking.enabled:
        return Accelerator(mixed_precision=config.system.mixed_precision)

    accelerator = Accelerator(mixed_precision=config.system.mixed_precision,
                              log_with=config.tracking.tool,
                              logging_dir=config.tracking.directory)

    # init trackers
    project_name = config.tracking.default_project_prefix
    tracking_root = os.path.join(os.path.dirname(__file__), config.tracking.directory)
    if config.tracking.project_name is None:
        version = 0
        while os.path.isdir(os.path.join(tracking_root, f'{project_name}_{version}')):
            # keep increment untill no collision
            version += 1

        project_name = f'{project_name}_{version}'
    else:
        project_name = re.sub('/', '_', config.tracking.project_name)

    accelerator.init_trackers(project_name)
    accelerator.tracking_dir = os.path.join(tracking_root, project_name)

    # save current configuration
    if accelerator.is_local_main_process:
        with open(os.path.join(accelerator.tracking_dir, 'config.yaml'), 'w') as f:
            f.write(config.dump())

    return accelerator


def set_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake detector with foundation models.")
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Optional YAML configuration file"
    )
    
    main(parser.parse_args().cfg)
