import json
import pickle
import argparse
import logging
from os import path, makedirs
from datetime import datetime

from accelerate import Accelerator
import evaluate
import torch
from torch.utils.data import DataLoader
from yacs.config import CfgNode as CN
from tqdm import tqdm
import yaml

from src.datasets import FFPP, CDF, DFDC
from src.models import Detector
from src.tools.notify import send_to_telegram


def get_config(cfg_file, args):
    with open(cfg_file) as f:
        preset = CN(yaml.safe_load(f))

    C = CN()

    # prerequisite: fetch the Deepfake detection task index during training.
    C.target_task = next(i for i, d in enumerate(preset.data.eval) if d.category == "Deepfake")

    if args.aux_file:
        with open(args.aux_file) as f:
            aux = CN(yaml.safe_load(f))

    C.data = CN()
    C.data.num_frames = preset.data.num_frames
    C.data.clip_duration = preset.data.clip_duration

    C.data.datasets = [
        globals()[d.name].get_default_config().merge_from_other_cfg(d)
        for d in preset.data.eval + (aux.data.eval if args.aux_file else [])
        if d.category == "Deepfake"
    ]

    if args.test:
        for cfg in C.data.datasets:
            cfg.scale = 0.1
    else:
        for cfg in C.data.datasets:
            cfg.scale = 1.0

    # model
    C.model = Detector.get_default_config().merge_from_other_cfg(preset.model)

    C.freeze()
    return C


def collate_fn(batch):
    # just forward the dataset
    # only work with batch_size=1
    return batch[0]


@torch.no_grad()
def main(args):
    root = args.artifacts_dir

    config = get_config(path.join(root, f"{args.cfg_name}.yaml"), args)

    accelerator = Accelerator()

    report = {}
    stats = {}

    for ds_cfg in config.data.datasets:
        # prepare model and dataset
        model = Detector(config.model, config.data.num_frames, accelerator).to(accelerator.device).eval()
        ds_cfg.pack = 1

        test_dataset = globals()[ds_cfg.name](
            ds_cfg,
            config.data.num_frames,
            config.data.clip_duration,
            model.transform,
            accelerator,
            split='test',
            index=config.target_task,
        )

        stats[ds_cfg.name] = {
            "label": [],
            "prob": []
        }
        test_dataloader = accelerator.prepare(DataLoader(
            test_dataset, num_workers=args.num_workers, collate_fn=collate_fn))
        logging.info(f'Dataset {test_dataset.__class__.__name__} initialized with {len(test_dataset)} samples\n')
        with accelerator.main_process_first():
            model.load_state_dict(torch.load(path.join(root, f'{args.weight_mode}_weights.pt')))

        if accelerator.is_local_main_process:
            accuracy_calc = evaluate.load("accuracy")
            roc_auc_calc = evaluate.load("roc_auc")

        N = args.batch_size
        progress_bar = tqdm(test_dataloader, disable=not accelerator.is_local_main_process)
        for i, data in enumerate(progress_bar):
            clips, label, masks, task_index = *data[:3], data[-1]
            if (len(clips) == 0):
                logging.error(f"Sample Index: {i} Cannot Provide Clip for Inference, Skipping...")
                continue
            logits = []
            for i in range(0, len(clips), N):
                logits.append(
                    model.predict(
                        torch.stack(clips[i:i + N]).to(accelerator.device),
                        torch.stack(masks[i:i + N]).to(accelerator.device)
                    )[0][task_index].detach().to("cpu")
                )

            p = torch.cat(logits).softmax(dim=-1)

            # means = p.mean(dim=0)

            # # ensemble option 1: greedly take the most confident score
            # pred_prob = (p[p[:, 1].argmax()] if means[1] > 0.5 # avg prob of real video is higher
            #         else p[p[:, 0].argmax()])[None]

            # # ensemble option 2: use the fakest score regardless of prediction
            # pred_prob = p[p[:, 0].argmax()][None]

            # ensemble option 3: just use the mean
            # pred_prob = means[None]

            if (args.modality == "clip"):
                pred_prob = p
                pred_label = pred_prob.argmax(dim=-1)
                label = torch.tensor(label, device="cpu")
            elif (args.modality == "video"):
                pred_prob = p.mean(dim=0).unsqueeze(0)
                pred_label = pred_prob.argmax(dim=-1).unsqueeze(0)
                label = torch.tensor(label[0], device="cpu").unsqueeze(0)
            else:
                raise NotImplementedError()

            # sync across process
            pred_probs, pred_labels, labels = accelerator.gather_for_metrics(
                (pred_prob, pred_label, label)
            )

            stats[ds_cfg.name]["label"] += labels.tolist()
            stats[ds_cfg.name]["prob"] += pred_probs[:, 1].tolist()

            if accelerator.is_local_main_process:
                accuracy_calc.add_batch(references=labels, predictions=pred_labels)
                roc_auc_calc.add_batch(references=labels, prediction_scores=pred_probs[:, 1])  # prob of real class

        if accelerator.is_local_main_process:
            accuracy_calc.add_batch(references=[0, 1], predictions=[0, 1])
            roc_auc_calc.add_batch(references=[0, 1], prediction_scores=[0, 1])  # prob of real class
            accuracy = round(accuracy_calc.compute()['accuracy'], 3)
            roc_auc = round(roc_auc_calc.compute()['roc_auc'], 3)
            logging.info(f'accuracy: {accuracy}, roc_auc: {roc_auc}')
            report[test_dataset.__class__.__name__] = {
                "accuracy": accuracy,
                "roc_auc": roc_auc
            }

    # save report and stats.
    timestamp = datetime.utcnow().strftime("%m%dT%H%M")
    with open(path.join(root, f'report_{timestamp}_{args.weight_mode}_{args.modality}.json'), "w") as f:
        json.dump(report, f, sort_keys=True, indent=4, separators=(',', ': '))

    with open(path.join(root, f'stats_{timestamp}_{args.weight_mode}_{args.modality}.pickle'), "wb") as f:
        pickle.dump(stats, f)

    send_to_telegram(f"Inference for '{root.split('/')[-1]}' Complete!")
    send_to_telegram(json.dumps(report, sort_keys=True, indent=4, separators=(',', ': ')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake detector with foundation models.")
    parser.add_argument(
        "artifacts_dir",
        type=str,
        help="Directory to optimized model artifacts"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--aux_file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--weight_mode",
        type=str,
        default="best",
    )

    parser.add_argument(
        "--modality",
        type=str,
        default="video"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8
    )

    parser.add_argument(
        "--test",
        action="store_true",
    )

    parser.add_argument(
        "--cfg_name",
        type=str,
        default="config"
    )

    main(parser.parse_args())
