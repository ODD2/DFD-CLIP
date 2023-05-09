import os
import sys
import random
import re
import argparse
import logging
from datetime import timedelta,datetime

import numpy as np
import torch
from accelerate import Accelerator,DistributedDataParallelKwargs
from yacs.config import CfgNode as CN

from src.models import Detector,CompInvEncoder
from src.datasets import RPPG,FFPP,DFDC,CDF
from src.trainer import Trainer,CompInvTrainer
from src.evaluator import Evaluator,CompInvEvaluator
from src.callbacks.timer import start_timer, end_timer
from src.callbacks.metrics import init_metrics, update_metrics, compute_metrics
from src.callbacks.tracking import update_trackers,  cache_best_model

def get_config(params):
    config_file = params.cfg
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
    C.tracking.directory = 'logs'
    C.tracking.project_name = None
    C.tracking.default_project_prefix = 'version'
    C.tracking.tool = 'wandb'
    C.tracking.main_metric = 'deepfake/ffpp/roc_auc' # accuracy | roc_auc
    C.tracking.compare_fn = 'max' # max | min

    # model
    C.model = CN(new_allowed=True)
    
    # trainer
    C.trainer = CN(new_allowed=True)

    # evaluator
    C.evaluator = CN(new_allowed=True)

    # data
    C.data = CN()
    C.data.num_frames = 50
    C.data.clip_duration = 10
    C.data.train = []
    C.data.eval = []

    # load additional configs from file
    if config_file is not None:
        C.merge_from_file(config_file)

        # model
        C.model = globals()[C.model.name].get_default_config().merge_from_other_cfg(C.model)

        # trainer
        C.trainer = globals()[C.trainer.name].get_default_config().merge_from_other_cfg(C.trainer)

        # evaluator
        C.evaluator = globals()[C.evaluator.name].get_default_config().merge_from_other_cfg(C.evaluator)
        
        # train dataset
        C.data.train = [
            (
                globals()[d_train.name]
                .get_default_config()
                .merge_from_other_cfg(d_train)
            )
            for d_train in C.data.train
        ]

        # eval datasets
        C.data.eval = [
            (
                globals()[d_eval.name]
                .get_default_config()
                .merge_from_other_cfg(d_eval)
            )
            for d_eval in C.data.eval
        ]

    if params.test:
        C.tracking.enabled = True
        C.tracking.directory = 'logs'
        C.tracking.project_name = "test"

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
        if kwargs['evaluator'].best_model_state:
            trainer.accelerator.save(kwargs['evaluator'].best_model_state,
                                    os.path.join(trainer.accelerator.project_dir, 'best_weights.pt'))
        if kwargs['evaluator'].last_model_state:
            trainer.accelerator.save(kwargs['evaluator'].last_model_state,
                                    os.path.join(trainer.accelerator.project_dir, 'last_weights.pt'))

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

    # stdout logger
    trainer.add_callback('on_batch_end',
        lambda trainer: trainer.accelerator.print(f'{trainer.steps} | loss {trainer.batch_loss_info}, {trainer.batch_duration:.2f}s'))
    trainer.add_callback('on_epoch_end',
        lambda trainer: trainer.accelerator.print(f'epoch takes {trainer.epoch_duration:.2f}s'))
    trainer.add_callback('on_training_end',
        lambda trainer: trainer.accelerator.print(f'training completed in {timedelta(seconds=trainer.training_duration)}'))

    # evaluator
    trainer.add_callback('on_batch_end', evaluation_proxy, evaluation_interval=config.system.evaluation_interval)


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
        evaluator.add_callback('on_evaluation_end', compute_metrics, training_eval_interval=1)

    # tracker
    if config.tracking.enabled and evaluator.accelerator.is_local_main_process:
        evaluator.add_callback('on_evaluation_end', update_trackers)

        # model saver
        evaluator.add_callback('on_evaluation_start', clear_current_main_metrics,
                                                      main_metric=config.tracking.main_metric,
                                                      compare_fn=config.tracking.compare_fn,
                                                      current_main_metrics=[])
        evaluator.add_callback('on_evaluation_end', cache_best_model, best_model_state=None, last_model_state=None)

    # stdout logger
    evaluator.add_callback('on_batch_end',
        lambda evaluator: evaluator.accelerator.print(f'{evaluator.steps}.{evaluator.batch_num} | loss {evaluator.batch_loss_info}')
    )

    evaluator.add_callback(
        'on_evaluation_end',
        lambda evaluator: evaluator.accelerator.print(f'evaluation completed in {evaluator.evaluation_duration:.2f}')
    )


def main(params):
    
    config = get_config(params)
    
    # initialize accelerator and trackers (if enabled)
    accelerator = init_accelerator(config)
    accelerator.print(config.dump())

    # set random seed for deterministic training
    if config.system.deterministic_training:
        set_seed(config.system.seed)

    # initialize model
    model =  globals()[config.model.name](config.model, accelerator=accelerator,num_frames=config.data.num_frames, )

    # category to task index mapping
    category_index = {
        cat: i for i,cat in enumerate(set([cfg.category for cfg in config.data.train]))
    }

    accelerator.print(f"Task Indices:")
    for k,v in category_index.items():
        accelerator.print(f"\t- {k} => {v}")

    # initialize datasets
    train_datasets = [
        globals()[cfg.name](
            cfg,
            config.data.num_frames,
            config.data.clip_duration,
            transform=model.transform,
            accelerator=accelerator,
            split='train',
            index=category_index[cfg.category]
        ) for cfg in config.data.train
    ]

    for dataset in train_datasets:
        accelerator.print(f'Training Dataset {dataset.__class__.__name__.upper()} initialized with {len(dataset)} samples\n')

    eval_datasets = [
        globals()[cfg.name](
            cfg, 
            config.data.num_frames,
            config.data.clip_duration,
            model.transform,
            accelerator=accelerator,
            split='val',
            index=category_index[cfg.category]
        ) for cfg in config.data.eval
    ]

    for dataset in eval_datasets:
        accelerator.print(f'Evaluation Dataset {dataset.__class__.__name__.upper()} initialized with {len(dataset)} samples\n')

    # initialize trainer and evaluator
    trainer = globals()[config.trainer.name](config.trainer, accelerator, model, train_datasets)
    evaluator =  globals()[config.evaluator.name](config.evaluator, accelerator, eval_datasets)

    # register callbacks
    register_trainer_callbacks(config, trainer, evaluator=evaluator)
    register_evaluator_callbacks(config, evaluator)

    # start training
    trainer.run()

    # finished
    if config.tracking.enabled:
        accelerator.end_training()


def init_accelerator(config):
    accelerate_kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    # init huggingface accelerator
    if not config.tracking.enabled:
        return Accelerator(mixed_precision=config.system.mixed_precision,kwargs_handlers=accelerate_kwargs_handlers)



    # init trackers
    project_name = config.tracking.default_project_prefix
    tracking_root = os.path.join(os.path.dirname(__file__), config.tracking.directory)
    if config.tracking.project_name is None:
        version = 0
        while os.path.isdir(os.path.join(tracking_root, f'{project_name}_{version}')):
            # keep increment untill no collision
            version += 1

        project_name = f'{project_name}_{version}'
        tracking_dir = os.path.join(tracking_root, project_name)
    else:
        project_name = re.sub('/', '_', config.tracking.project_name)
        tracking_dir = os.path.join(tracking_root, project_name,datetime.utcnow().strftime("%m%dT%H%M"))

    accelerator = Accelerator(
        mixed_precision=config.system.mixed_precision,
        log_with=config.tracking.tool,
        project_dir=tracking_dir,
        kwargs_handlers=accelerate_kwargs_handlers
    )

    accelerator.init_trackers(project_name)
    
    os.makedirs(accelerator.project_dir,exist_ok=True)

    # save current configuration
    if accelerator.is_local_main_process:
        with open(os.path.join(accelerator.project_dir, 'config.yaml'), 'w') as f:
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debugging Mode"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Testing Mode"
    )

    params = parser.parse_args()

    if(not params.debug):
        import warnings
        logging.basicConfig(level="INFO")
        warnings.filterwarnings(action="ignore")
        # warnings.simplefilter(action='ignore', category=RuntimeWarning)
    else:
        
        logging.basicConfig(level="DEBUG")
        

    main(params)
