import os
import sys
import random
import re
import argparse
import logging
from datetime import timedelta, datetime

import wandb
import torch
import numpy as np
from accelerate import Accelerator, DistributedDataParallelKwargs
from yacs.config import CfgNode as CN
from accelerate.utils import GradientAccumulationPlugin


from src.models import Detector, VPT, DetectorEVL
from src.datasets import RPPG, FFPP, DFDC, CDF, FSh
from src.trainer import Trainer, VPTTrainer
from src.evaluator import Evaluator, VPTEvaluator
from src.callbacks.timer import start_timer, end_timer
from src.callbacks.metrics import init_metrics, update_metrics, compute_metrics
from src.callbacks.tracking import update_trackers, cache_best_model
from src.tools.notify import send_to_telegram

from inference import main as inference_runner
from inference import parse_args as inference_arg_parser


PROJECT_DIR = None


def get_config(params):
    config_file = params.cfg
    C = CN()

    # system
    C.system = CN()
    C.system.mixed_precision = 'no'  # no | fp16 | bf16
    C.system.seed = 0
    C.system.deterministic_training = False
    C.system.training_eval_interval = 10
    C.system.evaluation_interval = 10
    C.system.evaluation_aggregate = 1

    # tracking
    C.tracking = CN()
    C.tracking.enabled = False
    C.tracking.directory = 'logs'
    C.tracking.project_name = None
    C.tracking.tool = 'wandb'
    C.tracking.main_metric = 'deepfake/ffpp/roc_auc'  # accuracy | roc_auc
    C.tracking.compare_fn = 'max'  # max | min

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
        # C.tracking.enabled = True
        C.tracking.directory = 'logs'
        C.tracking.project_name = "test"

    if params.debug:
        C.tracking.enabled = False
        C.trainer.num_workers = 0
        C.trainer.batch_size = 1
        C.trainer.batch_accum = 1

    # some sanity check
    for d_eval in C.data.eval:
        assert 'name' in d_eval
    C.freeze()

    return C


def register_trainer_callbacks(config, trainer, evaluator):
    def evaluation_proxy(trainer):
        if trainer.steps % config.system.evaluation_interval:
            return
        # best main metric is none before the first evaluation
        prev_metric = getattr(evaluator, 'best_main_metric', None)
        evaluator.run(trainer)
        next_metric = getattr(evaluator, 'best_main_metric', None)

        if trainer.steps % (config.system.evaluation_interval * config.system.evaluation_aggregate):
            return

        # record
        if (prev_metric == next_metric):
            trainer.early_stop_counter += 1
        else:
            trainer.early_stop_counter = 0

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
        trainer.add_callback(
            'on_batch_end',
            compute_metrics,
            training_eval_interval=config.system.training_eval_interval
        )

    # tracker
    if config.tracking.enabled and trainer.accelerator.is_local_main_process:
        trainer.add_callback('on_batch_end', update_trackers)
        # TODO: save training status
        # trainer.add_callback('on_training_end', save_training_status)

    # stdout logger
    trainer.add_callback(
        'on_batch_end',
        lambda trainer: logging.info(
            f'{trainer.steps} | loss {trainer.batch_loss_info}, {trainer.batch_duration:.2f}s'
        )
    )
    trainer.add_callback(
        'on_epoch_end',
        lambda trainer: logging.info(
            f'epoch takes {trainer.epoch_duration:.2f}s'
        )
    )
    trainer.add_callback(
        'on_training_end',
        lambda trainer: logging.info(
            f'training completed in {timedelta(seconds=trainer.training_duration)}'
        )
    )

    # evaluator
    trainer.add_callback(
        'on_batch_end',
        evaluation_proxy
    )


def register_evaluator_callbacks(config, evaluator, trainer):
    def save_model(evaluator):
        if evaluator.best_model_state:
            evaluator.accelerator.save(
                evaluator.best_model_state,
                os.path.join(PROJECT_DIR, 'best_weights.pt')
            )
        if evaluator.last_model_state:
            evaluator.accelerator.save(
                evaluator.last_model_state,
                os.path.join(PROJECT_DIR, 'last_weights.pt')
            )

    def evaluation_aggregator(fn, end=False):
        # mode:
        #   True - executes only first step
        #   False - executes only the last step.
        eval_interval = config.system.evaluation_interval
        eval_aggregate = config.system.evaluation_aggregate

        # just execute each step
        if eval_aggregate == 1:
            return fn

        # execute in two conditions
        if not end:
            def driver(evaluator):
                assert trainer.steps % eval_interval == 0
                eval_count = trainer.steps // eval_interval
                if (eval_count % eval_aggregate == 1):
                    fn(evaluator)
        else:
            def driver(evaluator):
                assert trainer.steps % eval_interval == 0
                eval_count = trainer.steps // eval_interval
                if (eval_count % eval_aggregate == 0):
                    fn(evaluator)

        return driver

    # timer
    timer_events = ['evaluation', 'dataloader']
    evaluator.add_callback('on_evaluation_start', lambda _: None, timer={evt: 0 for evt in timer_events})
    evaluator.add_callback('on_evaluation_start', lambda evaluator: logging.info('evaluation start'))
    for event in timer_events:
        evaluator.add_callback(f'on_{event}_start', start_timer)
        evaluator.add_callback(f'on_{event}_end', end_timer)

    # metrics & tracker
    evaluator.add_callback('on_batch_end', update_metrics)
    if evaluator.accelerator.is_local_main_process:
        def evalution_end_procedures(evaluator):
            # metric computation
            compute_metrics(evaluator)
            # model saver
            cache_best_model(evaluator)
            save_model(evaluator)

        # metric initialization
        evaluator.add_callback(
            'on_evaluation_start',
            evaluation_aggregator(init_metrics, end=False),
        )

        evaluator.add_callback(
            'on_evaluation_end',
            evaluation_aggregator(evalution_end_procedures, end=True),
            training_eval_interval=1,
            main_metric=config.tracking.main_metric,
            compare_fn=config.tracking.compare_fn,
            best_model_state=None,
            last_model_state=None
        )

    # stdout logger
    evaluator.add_callback(
        'on_batch_end',
        lambda evaluator: logging.info(
            f'{evaluator.steps}.{evaluator.batch_num} | loss {evaluator.batch_loss_info}'
        )
    )

    evaluator.add_callback(
        'on_evaluation_end',
        lambda evaluator: logging.info(
            f'evaluation completed in {evaluator.evaluation_duration:.2f}'
        )
    )


def main(params):
    global PROJECT_DIR

    config = get_config(params)

    # initialize accelerator and trackers
    accelerator = init_accelerator(config, params)
    logging.info(config.dump())

    # set random seed for deterministic training
    if config.system.deterministic_training:
        set_seed(config.system.seed)

    # initialize model
    model = globals()[config.model.name](config.model, accelerator=accelerator, num_frames=config.data.num_frames, )

    # watch models
    wandb.watch(model, log="all", log_freq=100)

    # category to task index mapping
    category_index = {
        cat: i for i, cat in enumerate(set([cfg.category for cfg in config.data.train]))
    }

    logging.info(f"Task Indices:{category_index}")

    # initialize datasets
    train_datasets = [
        globals()[cfg.name](
            cfg,
            config.data.num_frames,
            config.data.clip_duration,
            transform=model.transform,
            accelerator=accelerator,
            split='train',
            n_px=model.encoder.input_resolution,
            index=category_index[cfg.category]
        ) for cfg in config.data.train
    ]

    for dataset in train_datasets:
        logging.info(
            f'Training Dataset {dataset.__class__.__name__.upper()} initialized with {len(dataset)} samples.'
        )

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
        logging.info(
            f'Evaluation Dataset {dataset.__class__.__name__.upper()} initialized with {len(dataset)} samples.'
        )

    # initialize trainer and evaluator
    trainer = globals()[config.trainer.name](config.trainer, accelerator, model, train_datasets)
    evaluator = globals()[config.evaluator.name](config.evaluator, accelerator, eval_datasets)

    # register callbacks
    register_trainer_callbacks(config, trainer, evaluator)
    register_evaluator_callbacks(config, evaluator, trainer)

    # start training
    trainer.run()

    # finished
    if config.tracking.enabled:
        wandb.finish()

    send_to_telegram(f"Training Completed, Result Location: {PROJECT_DIR}")


def init_accelerator(config, params):
    global PROJECT_DIR

    accelerate_kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(
        mixed_precision=config.system.mixed_precision,
        gradient_accumulation_steps=config.trainer.batch_accum,
        kwargs_handlers=accelerate_kwargs_handlers
    )

    # init tracker parameters
    tracking_root = os.path.join(os.path.dirname(__file__), config.tracking.directory)

    # resolve project name
    if config.tracking.project_name is None:
        config.tracking.project_name = "temp"
    project_name = re.sub('/', '_', config.tracking.project_name)

    # init tracker.
    wandb.init(project=project_name, mode="online" if config.tracking.enabled else "offline")

    if (wandb.run.name == None):
        wandb.run.name = datetime.utcnow().strftime("%m%dT%H%M")

    # set project directory
    PROJECT_DIR = os.path.join(tracking_root, project_name, wandb.run.name)

    # save current configuration
    os.makedirs(PROJECT_DIR, exist_ok=True)
    if accelerator.is_local_main_process:
        with open(os.path.join(PROJECT_DIR, 'setting.yaml'), 'w') as f:
            f.write(config.dump())

    # upload config.
    wandb.save(glob_str=os.path.join(PROJECT_DIR, 'setting.yaml'), policy="now")

    # save run name in description.
    wandb.run.notes = f"{wandb.run.name}:{params.notes}"

    return accelerator


def set_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake detector with foundation models.")
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Optional YAML configuration file"
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="no description",
        help="Optional message to update the notes in wandb."
    )

    parser.add_argument(
        "--aux_file",
        type=str,
        default="configs/inference/all.yaml",
        help="The configuration for inference after training."
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

    logging_fmt = "[%(levelname)s][%(filename)s:%(lineno)d]: %(message)s"

    if (not params.debug):
        import warnings
        logging.basicConfig(level="INFO", format=logging_fmt)
        warnings.filterwarnings(action="ignore")
        # disable warnings from the xformers efficient attention module due to torch.user_deterministic_algorithms(True,warn_only=True)
        warnings.filterwarnings(
            action="ignore",
            message=".*efficient_attention_forward_cutlass.*",
            category=UserWarning
        )
    else:
        logging.basicConfig(level="DEBUG", format=logging_fmt)

    # train model
    main(params)

    # run inference after training
    inference_runner(
        inference_arg_parser([
            PROJECT_DIR, params.aux_file
        ])
    )
