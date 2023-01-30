import argparse
from os import path, makedirs
import pickle

from accelerate import Accelerator
import  evaluate
import torch
from torch.utils.data import DataLoader 
from yacs.config import CfgNode as CN
from tqdm import tqdm
import yaml

from src.datasets import FFPP
from src.models import Detector


def get_config(cfg_file):
    with open(cfg_file) as f:
        preset = CN(yaml.safe_load(f))

    C = CN()
    # data
    C.data = [FFPP.get_default_config().merge_from_other_cfg(d) for d in preset.data.eval]

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
    config = get_config(path.join(root, 'config.yaml'))
    accelerator = Accelerator()

    for dataset in config.data:
        # prepare model and dataset
        model = Detector(config.model, dataset.num_frames, accelerator).to(accelerator.device).eval()
        test_dataset = FFPP(dataset, model.transform, accelerator, 'test')
        test_dataloader = accelerator.prepare(DataLoader(test_dataset, collate_fn=collate_fn))
        accelerator.print(f'Dataset {test_dataset.name} initialized with {len(test_dataset)} samples\n')
        with accelerator.main_process_first():
            model.load_state_dict(torch.load(path.join(root, 'best_weights.pt')))
        
        if accelerator.is_local_main_process:
            accuracy_calc = evaluate.load("accuracy")
            roc_auc_calc = evaluate.load("roc_auc")

        N = args.batch_size
        progress_bar = tqdm(test_dataloader, disable=not accelerator.is_local_main_process)
        for clips, label, masks in progress_bar:
            logits = []
            for i in range(0, len(clips), N):
                logits.append(model.predict(torch.stack(clips[i:i+N]).to(accelerator.device),
                                            torch.stack(masks[i:i+N]).to(accelerator.device)))
            label = torch.tensor([label], device=accelerator.device)

            p = torch.cat(logits).softmax(dim=-1)
            means = p.mean(dim=0)

            # # ensemble option 1: greedly take the most confident score
            # pred_prob = (p[p[:, 1].argmax()] if means[1] > 0.5 # avg prob of real video is higher
            #         else p[p[:, 0].argmax()])[None]

            # # ensemble option 2: use the fakest score regardless of prediction
            # pred_prob = p[p[:, 0].argmax()][None]

            # ensemble option 3: just use the mean
            pred_prob = means[None]

            pred_label = pred_prob.argmax(dim=-1)

            # sync across process
            pred_probs, pred_labels, labels = accelerator.gather_for_metrics((
            pred_prob, pred_label, label))

            if accelerator.is_local_main_process:
                accuracy_calc.add_batch(references=labels, predictions=pred_labels)
                roc_auc_calc.add_batch(references=labels, prediction_scores=pred_probs[:, 1]) # prob of real class

        if accelerator.is_local_main_process:
            accuracy = accuracy_calc.compute()['accuracy']
            roc_auc = roc_auc_calc.compute()['roc_auc']
            print(f'accuracy: {accuracy}, roc_auc: {roc_auc}')


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
        default=16,
    )
    
    main(parser.parse_args())

