import os
import yaml
import pickle
import random
import logging
import numpy as np
from math import comb
from enum import auto, IntEnum
from itertools import combinations
from collections import OrderedDict

from yacs.config import CfgNode as CN
import torch
from torch import nn
from . import clip
import torchvision.transforms as T


def remap_weight(
        raw_weights,
        model=None
):

    weights = {}
    for k in raw_weights:
        if (type(model) == Detector):
            if k == "decoder.ln_post.weight":
                weights[f"decoder.proj0x2_L{model.layer_indices[-1]}.0.weight"] = raw_weights[k]
            elif k == "decoder.ln_post.bias":
                weights[f"decoder.proj0x2_L{model.layer_indices[-1]}.0.bias"] = raw_weights[k]
            elif k == "decoder.proj0x2":
                weights[f"decoder.proj0x2_L{model.layer_indices[-1]}.2.weight"] = raw_weights[k].T
            elif k == "decoder.class_embedding" and len(raw_weights[k].shape) == 1:
                weights[k] = raw_weights[k].unsqueeze(0)
            else:
                weights[k] = raw_weights[k]
        else:
            if k == "prompt_embeddings":
                weights[f"frame_prompt_embeddings"] = raw_weights[k]
            else:
                weights[k] = raw_weights[k]

    return weights


class AttnMode(IntEnum):
    PATCH = auto()
    FRAME = auto()
    TEMPORAL = auto()


class Foundations(IntEnum):
    CLIP = auto()
    DINO2 = auto()
    FARL = auto()


class DecodeMode(IntEnum):
    INDEX = auto()
    STRIDE = auto()


class AdaptorMode(IntEnum):
    PRETRAIN = auto()
    NORMAL = auto()
    NONE = auto()


class TemporalMetric(IntEnum):
    RANK = auto()
    TRIPLET = auto()
    NONE = auto()


class PatchMaskMode(IntEnum):
    BATCH = auto()
    SAMPLE = auto()
    GUIDE = auto()
    NONE = auto()


class CompMetric(IntEnum):
    FEATURE = auto()
    SYNC = auto()
    NONE = auto()


class QueryGuideMode(IntEnum):
    NORMAL = auto()
    NONE = auto()


class Optimizers(IntEnum):
    SGD = auto()
    ADAMW = auto()


class Object(object):
    pass


def auc_roc(weight=None, label_smoothing=0.0, *args, **kargs):
    def driver(logits, y, _weight=weight, _label_smoothing=label_smoothing):
        if (_weight):
            _weight = torch.tensor(_weight, device=logits.device)
        return torch.nn.functional.cross_entropy(
            logits,
            y,
            weight=_weight,
            label_smoothing=_label_smoothing,
            reduction='none'
        )
    return driver


def enable_gradients(module: nn.Module):
    for params in module.parameters():
        params.requires_grad = True
    return module


def disable_gradients(module: nn.Module):
    for params in module.parameters():
        params.requires_grad = False
    return module


class LayerNorm(nn.LayerNorm):
    """
    Subclass torch's LayerNorm to handle fp16.
    Ported from:
    https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L157
    """

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """
    Ported from:
    https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L166
    """

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MultiheadAttention(nn.Module):
    """
    The modified multihead attention do not project k and v,
    it accepts k, v exported from CLIP ViT layers,

    q is learnable and it attend to ported k, v but not itself (no self attension)

    frames that do not exist at the end of a video are masked out
    to prevent q attends to the pads

    in addition to the original softmax in the original transformer
    we also apply compositional de-attention to the affinity matrix
    """

    def __init__(self, config, num_frames, embed_dim, n_head):
        super().__init__()
        self.num_frames = num_frames
        self.attn_mode = config.op_mode.attn_mode
        self.attn_record = config.op_mode.attn_record

        def smax(q, k, m):
            """
            softmax in the original Transformer
            """
            aff = torch.einsum('nqhc,nkhc->nqkh', q / (q.size(-1) ** 0.5), k)
            aff = aff.masked_fill(~m, float('-inf'))
            n, q, k, h = aff.shape
            affinities = []

            if AttnMode.PATCH in self.attn_mode:
                logging.debug("perform patch mode attention.")
                affinities.append(aff.softmax(dim=-2))

            if AttnMode.FRAME in self.attn_mode:
                logging.debug("perform frame mode attention.")
                affinities.append(aff.view((n, q, self.num_frames, -1, h)).softmax(dim=-2).view((n, q, k, h)))

            if AttnMode.TEMPORAL in self.attn_mode:
                logging.debug("perform temporal mode attention.")
                affinities.append(aff.view((n, q, self.num_frames, -1, h)).softmax(dim=-3).view((n, q, k, h)))

            return sum(affinities) / len(affinities)

        def coda(q, k, m):
            """
            Compositional De-Attention Networks, Neurips 19
            """
            norm = q.size(-1) ** 0.5
            aff = torch.einsum('nqhc,nkhc->nqkh', q / norm, k).tanh()
            gate = -(q.unsqueeze(2) - k.unsqueeze(1)).abs().sum(-1) / norm
            gate = 2 * gate.sigmoid().masked_fill(~m, 0.)
            return aff * gate

        # create list of activation drivers.
        self.activations = []
        for driver in config.op_mode.attn_driver:
            self.activations.append(locals()[driver])

        self.n_act = len(self.activations)  # softmax and coda
        self.in_proj = nn.Linear(embed_dim, self.n_act * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.embed_dim = embed_dim
        self.n_head = n_head

        self.aff = None
        self.qs = None

    def forward(self, q, k, v, m):
        # qs is a tuple with elements with shape equal to: b, q, h, d
        qs = self.in_proj(q).view(*q.shape[:2], self.n_head, -1).split(self.embed_dim // self.n_head, -1)
        m = m.unsqueeze(1).unsqueeze(-1)

        aff = 0
        for i in range(self.n_act):
            aff += self.activations[i](qs[i], k, m) / self.n_act

        if self.attn_record:
            logging.debug("recording attention results.")
            self.aff = aff
            self.qs = qs

        mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)

        return self.out_proj(mix.flatten(-2)), torch.stack([q.flatten(-2) for q in qs], dim=1)


class ResidualAttentionBlock(nn.Module):
    """
    Modified from:
    https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L171
    """

    def __init__(self, d_model: int, n_head: int, config, num_frames, block_index, layer_indices, reference_layers):
        super().__init__()

        self.attn = MultiheadAttention(config, num_frames, d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(config.dropout)),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

        self._apply_reference(config, block_index, layer_indices, reference_layers)
        self.ln_1.requires_grad_(False)
        self.mlp.requires_grad_(False)
        self.ln_2.requires_grad_(False)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, m: torch.Tensor):
        return self.attn(q, k, v, m)

    def forward(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor, m: torch.Tensor):
        _x, qs = self.attention(self.ln_1(x), k, v, m)
        x = x + _x
        x = x + self.mlp(self.ln_2(x))
        return x, qs

    def _apply_reference(self, config, block_index, layer_indices, reference_layers):

        if config.foundation == Foundations.CLIP or config.foundation == Foundations.FARL:
            """
                use CLIP weights to initialize decoder
            """
            def fetch_ln1_params(layer_idx):
                return reference_layers[layer_idx].ln_1.state_dict()

            def fetch_ln2_params(layer_idx):
                return reference_layers[layer_idx].ln_2.state_dict()

            def fetch_mlp_params(layer_idx):
                return reference_layers[layer_idx].mlp.state_dict()

        elif config.foundation == Foundations.DINO2:
            """
                use DINO2 weights to initialize decoder
            """
            def fetch_ln1_params(layer_idx):
                return reference_layers[layer_idx].norm1.state_dict()

            def fetch_ln2_params(layer_idx):
                return reference_layers[layer_idx].norm2.state_dict()

            def fetch_mlp_params(layer_idx):
                weights = reference_layers[layer_idx].mlp.state_dict()
                name_convert = {"fc1": "c_fc", "fc2": "c_proj"}
                mapped_weights = {}
                for k, v in weights.items():
                    mname = k.split('.')[0]
                    if mname in name_convert:
                        k = k.replace(mname, name_convert[mname])
                    mapped_weights[k] = v

                return mapped_weights
        else:
            raise NotImplementedError()

        logging.debug("perform normal reference initialization.")
        current_layer = layer_indices[block_index]
        self.ln_1.load_state_dict(fetch_ln1_params(current_layer))
        self.mlp.load_state_dict(fetch_mlp_params(current_layer))
        self.ln_2.load_state_dict(fetch_ln2_params(current_layer))


class Transformer(nn.Module):
    """
    Modified from:
    https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L195
    """

    def __init__(self, width: int, heads: int, config, num_frames, layer_indices, reference_layers):
        super().__init__()
        self.config = config
        self.width = width
        self.resblocks = []
        for block_index in range(len(layer_indices)):
            self.resblocks.append(
                ResidualAttentionBlock(
                    width, heads, config, num_frames,
                    block_index, layer_indices, reference_layers
                )
            )

        self.resblocks = nn.Sequential(*self.resblocks)

    def forward(self, x: torch.Tensor, kvs, m):
        layer_results = []
        layer_qs = []
        for i, blk, kv in zip(range(len(self.resblocks)), self.resblocks, kvs):
            x, qs = blk(x, kv['k'], kv['v'], m)
            layer_results.append(x.unsqueeze(1))
            layer_qs.append(qs.unsqueeze(1))

        return torch.cat(layer_results, dim=1), torch.cat(layer_qs, dim=1)


class Decoder(nn.Module):
    """
    The decoder aggregates the keys and values exported from CLIP ViT layers
    and predict the truthfulness of a video clip

    The positional embeddings are shared across patches in the same spatial location
    """

    def __init__(self, detector, config, num_frames):
        super().__init__()
        self.config = config
        width = detector.encoder.width
        heads = detector.encoder.heads
        output_dims = config.out_dim
        scale = width ** -0.5
        self.num_classes = config.num_classes
        self.class_embedding = nn.Parameter(scale * torch.randn(self.num_classes, width))

        self.positional_embedding = nn.Parameter(scale * torch.randn(num_frames, 1, heads, width // heads))

        self.ln_pre = LayerNorm(width)
        self.drop_pre = torch.nn.Dropout(config.dropout)
        self.transformer = Transformer(
            width,
            heads,
            config,
            num_frames,
            layer_indices=detector.layer_indices,
            reference_layers=detector.encoder.transformer.resblocks
        )

        self.task_projections = []
        for i, output_dim in enumerate(output_dims):
            _name = f"proj{i}x{output_dim}_L{detector.layer_indices[-1]}"
            setattr(
                self,
                _name,
                nn.Sequential(
                    LayerNorm(width),
                    nn.Dropout(config.dropout),
                    nn.Linear(width, output_dim, bias=False)
                )
            )
            self.task_projections.append(getattr(self, _name))

    def forward(self, kvs, m):
        m = m.repeat_interleave(kvs[0]['k'].size(2), dim=-1)
        # add temporasitional embedding
        logging.debug("perform temporal position embedding.")
        for i in range(len(kvs)):
            for k in kvs[i].keys():
                kvs[i][k] = kvs[i][k] + self.positional_embedding

        # flatten
        for i in range(len(kvs)):
            for k in kvs[i].keys():
                kvs[i][k] = kvs[i][k].flatten(1, 2)

        x = self.class_embedding.unsqueeze(0).repeat(kvs[0]['k'].size(0), 1, 1)
        x = self.drop_pre(self.ln_pre(x))

        layer_results, layer_qs = self.transformer(
            x,
            kvs,
            m
        )

        x = layer_results.mean(dim=2)

        # drop the layer features if not required
        video_feature = x[:, -1].squeeze(1)

        task_logits = [
            task_heads(video_feature)
            for task_heads in self.task_projections
        ]

        return task_logits, video_feature, layer_results, layer_qs


class DINOv2(nn.Module):
    def __init__(self):
        super().__init__()
        from dinov2.models.vision_transformer import vit_base
        self.backbone = vit_base(img_size=518, patch_size=14, block_chunks=0, init_values=1.0, ffn_layer="mlp")
        self.backbone.load_state_dict(torch.load("misc/dinov2_vitb14_pretrain.pth"))
        # interfaces
        self.transformer = Object()
        self.transformer.resblocks = self.backbone.blocks
        # parameters
        self.heads = 12
        self.width = 768
        self.input_resolution = 224
        self.patch_size = 14
        self.block_num = len(self.transformer.resblocks)

    def forward(self, x, feat_keys=["k", "v"], **args):
        ret = self.backbone(x, is_training=True, **args)
        ret = {k: ret[k] for k in feat_keys}
        # convert {k1:[x1 ... xN], k2:[x1...xN]} -> [{k1[1],k2[1]} ... {k1[N],k2[N]}]
        ret = [
            {
                k: ret[k][i]
                for k in ret
            }
            for i in range(self.block_num)
        ]
        return ret


class Detector(nn.Module):
    """
    Deepfake video detector (also a video classifier)

    A series of video frames are first fed into CLIP image encoder
    we export key and values for each frame in each layer of CLIP ViT
    these k, v then further flatten on the temporal dimension
    resulting in a series of tokens for each video.

    The decoder aggregates the exported key and values in an attention manner 
    a CLS query token is learned during decoding.
    """
    @staticmethod
    def get_default_config():
        C = CN()
        C.name = "Detector"
        C.foundation = Foundations.CLIP.name.lower()
        C.architecture = 'ViT-B/16'

        # number of prompts
        C.frame_prompts = 0
        C.prompt_mode = ""
        C.prompt_layers = -1

        # number of classes
        C.num_classes = 1

        # decode mode
        C.decode_mode = CN()
        C.decode_mode.type = DecodeMode.INDEX.name.lower()
        # > argument for INDEX mode
        C.decode_mode.indices = []
        # > argument for STRIDE mode
        C.decode_mode.stride = -1

        # output dimension, for multiple tasks
        C.out_dim = []

        # output losses
        C.losses = []

        # training mode configurations
        C.train_mode = CN()
        # > patch mask
        C.train_mode.patch_mask = CN()
        C.train_mode.patch_mask.type = PatchMaskMode.NONE.name.lower()
        # - argument for GUIDED masking mode.
        C.train_mode.patch_mask.path = ""
        # - unmask patch ratio
        C.train_mode.patch_mask.ratio = 1.0
        # > query guide
        C.train_mode.query_guide = CN()
        C.train_mode.query_guide.type = QueryGuideMode.NONE.name.lower()
        # - for normal mode
        C.train_mode.query_guide.path = ""
        C.train_mode.query_guide.key = ""
        C.train_mode.query_guide.parts = []

        # operation mode configurations
        C.op_mode = CN()
        # > smax attention mode
        C.op_mode.attn_mode = AttnMode.PATCH.name.lower()
        # > attention activations
        C.op_mode.attn_driver = ["smax", "coda"]
        # > record attention scores
        C.op_mode.attn_record = False
        # > operate with frame modality
        C.op_mode.frame_task = False

        # regularization
        C.dropout = 0.0
        C.weight_decay = 0.01

        # optimizer
        C.optimizer = Optimizers.SGD.name.lower()

        return C

    @staticmethod
    def validate_config(config):
        config = config.clone()
        config.defrost()

        config.foundation = int(Foundations[config.foundation.upper()])
        assert config.foundation == Foundations.CLIP

        assert type(config.frame_prompts) == int
        assert config.frame_prompts >= 0
        assert type(config.prompt_mode) == str
        assert type(config.prompt_layers) == int
        assert not (config.frame_prompts > 0) or config.prompt_layers == -1 or config.prompt_layers > 0

        assert type(config.num_classes) == int and config.num_classes > 0

        config.decode_mode.type = int(DecodeMode[config.decode_mode.type.upper()])
        if config.decode_mode.type == DecodeMode.STRIDE:
            assert type(config.decode_mode.stride) == int
            assert config.decode_mode.stride > 0, "invalid value for decode stride."
        elif config.decode_mode.type == DecodeMode.INDEX:
            assert type(config.decode_mode.indices) == list
            assert len(config.decode_mode.indices) > 0, "indices unspecified for decoding."

        assert type(config.out_dim) == list

        assert type(config.losses) == list

        config.train_mode.patch_mask.type = int(PatchMaskMode[config.train_mode.patch_mask.type.upper()])
        assert type(config.train_mode.patch_mask.ratio) == float
        assert 0 < config.train_mode.patch_mask.ratio <= 1
        if config.train_mode.patch_mask.type == PatchMaskMode.GUIDE:
            assert type(config.train_mode.patch_mask.path) == str
            assert len(config.train_mode.patch_mask.path) > 0

        config.train_mode.query_guide.type = int(
            QueryGuideMode[config.train_mode.query_guide.type.upper()]
        )
        if config.train_mode.query_guide.type == QueryGuideMode.NORMAL:
            assert len(config.train_mode.query_guide.path) > 0
            assert len(config.train_mode.query_guide.key) > 0
            assert len(config.train_mode.query_guide.parts) > 0

        config.op_mode.attn_mode = [int(AttnMode[opt.upper()]) for opt in config.op_mode.attn_mode.split("+")]
        assert type(config.op_mode.attn_driver) == list
        assert type(config.op_mode.attn_record) == bool

        assert type(config.op_mode.frame_task) == bool
        assert not config.op_mode.frame_task or config.frame_prompts > 0

        assert type(config.dropout) == float
        assert 0 <= config.dropout <= 1

        assert type(config.weight_decay) == float
        assert 0 <= config.weight_decay <= 1

        config.optimizer = int(Optimizers[config.optimizer.upper()])

        config.freeze()
        return config

    def __init__(self, config, num_frames, accelerator):
        super().__init__()
        config = self.validate_config(config)
        self.config = config

        with accelerator.main_process_first():
            if config.foundation == Foundations.CLIP:
                self.encoder = clip.load(
                    config.architecture,
                    prompt_mode=config.prompt_mode,
                    frame_prompts=config.frame_prompts,
                    prompt_layers=config.prompt_layers,
                    attn_record=config.op_mode.attn_record
                )[0].visual.float()
            elif config.foundation == Foundations.DINO2:
                self.encoder = DINOv2()
            elif config.foundation == Foundations.FARL:
                _model = clip.load("ViT-B/16")[0]
                _model.load_state_dict(
                    torch.load("./misc/FaRL-Base-Patch16-LAIONFace20M-ep64.pth")["state_dict"],
                    strict=False
                )
                self.encoder = _model.visual.float()

        self.encoder = disable_gradients(self.encoder)
        self.encoder.frame_prompt_embeddings.requires_grad_(True)

        self.losses = []

        for loss in config.losses:
            if type(loss) == str:
                self.losses.append(globals()[loss]())
            else:
                self.losses.append(globals()[loss.name](**(dict(loss.args) if "args" in loss else {})))

        if (config.decode_mode.type == DecodeMode.STRIDE):
            self.layer_indices = list(range(0, len(self.encoder.transformer.resblocks), config.decode_mode.stride))
        elif (config.decode_mode.type == DecodeMode.INDEX):
            self.layer_indices = config.decode_mode.indices

        self.decoder = Decoder(self, config, num_frames)

        # transformations
        self.transform = self._transform(self.encoder.input_resolution)

        # image specific modules
        if self.config.op_mode.frame_task:
            self.task_projections = []
            for i, output_dim in enumerate(config.out_dim):
                _name = f"proj{i}x{output_dim}"
                setattr(
                    self,
                    _name,
                    nn.Sequential(
                        LayerNorm(self.encoder.width),
                        nn.Dropout(config.dropout),
                        nn.Linear(self.encoder.width, output_dim, bias=False)
                    )
                )
                self.task_projections.append(getattr(self, _name))

        if (self.config.train_mode.patch_mask.type == PatchMaskMode.GUIDE):
            with open(self.config.train_mode.patch_mask.path, "rb") as f:
                self.guide_map = pickle.load(f)

        if (self.config.train_mode.query_guide.type == QueryGuideMode.NORMAL):
            file_path = self.config.train_mode.query_guide.path
            parts = self.config.train_mode.query_guide.parts
            key = self.config.train_mode.query_guide.key
            with open(file_path, "rb") as f:
                prefetch_queries = pickle.load(f)
                self.video_semantic_queries = []
                for i in self.layer_indices:
                    layer_results = []
                    for part in parts:
                        layer_results.append(prefetch_queries[key][part][i])
                    self.video_semantic_queries.append(torch.stack(layer_results))
                self.video_semantic_queries = torch.stack(self.video_semantic_queries)
                self.video_semantic_queries.requires_grad_(False)

    def predict(self, x, m, with_video_features=False, train=False):
        b, t, c, h, w = x.shape
        features = {}

        # get key and value from each CLIP ViT layer
        kvs = self.encoder(x.flatten(0, 1), with_q=True, with_out=True, with_prompt=True)

        # restore temporal dimension
        for i in range(len(kvs)):
            for k in kvs[i]:
                for spec in kvs[i][k]:
                    if not type(kvs[i][k][spec]) == type(None):
                        kvs[i][k][spec] = kvs[i][k][spec].unflatten(0, (b, t))

        # save raw kvs video features for further usage
        features["kvs"] = [{k1: {k2: kv[k1][k2] for k2 in kv[k1].keys()} for k1 in kv.keys()} for kv in kvs]

        # discard prompt features and reserve only the origin.
        for i in range(len(kvs)):
            for k in kvs[i]:
                kvs[i][k] = kvs[i][k]["origin"]

        # the frame modality training
        if self.config.op_mode.frame_task:
            # prefetch image task logits
            image_logits = [
                proj(kvs[-1]["out"][:, 0, 0])
                for proj in self.task_projections
            ]
        else:
            image_logits = None

        # discard the 'out's
        for i in range(len(kvs)):
            for k in list(kvs[i].keys()):
                if not k in ["k", "v"]:
                    kvs[i].pop(k)

        # discard CLS token & prompts
        for i in range(len(kvs)):
            for k in kvs[i]:
                kvs[i][k] = kvs[i][k][:, :, 1:]

        # discard unwanted layers
        kvs = [kvs[i] for i in self.layer_indices]

        if train and not self.config.train_mode.patch_mask.type == PatchMaskMode.NONE:
            logging.debug("enter patch masking block.")
            patch_indices = None
            num_patch = kvs[0]["k"].shape[2]
            num_select = int(num_patch * self.config.train_mode.patch_mask.ratio)

            # partially discard patch kv pairs
            for i in range(len(kvs)):
                if self.config.train_mode.patch_mask.type == PatchMaskMode.BATCH:
                    logging.debug("perform batch patch masking.")
                    # discard patch localized out of the "batch" selected indices.
                    if type(patch_indices) == type(None):
                        patch_indices = np.random.choice(
                            range(num_patch),
                            num_select,
                            replace=False
                        )
                elif self.config.train_mode.patch_mask.type == PatchMaskMode.SAMPLE:
                    logging.debug("perform sample patch masking.")
                    # discard patch localized out of the randomly selected indices.
                    patch_indices = np.random.choice(
                        range(num_patch),
                        num_select,
                        replace=False
                    )
                elif self.config.train_mode.patch_mask.type == PatchMaskMode.GUIDE:
                    logging.debug("perform guide patch masking.")
                    patch_indices = np.random.choice(
                        range(num_patch),
                        num_select,
                        replace=False,
                        p=self.guide_map['v'][self.layer_indices[i]].flatten()
                    )

                for k in kvs[i]:
                    kvs[i][k] = kvs[i][k][:, :, patch_indices]

        video_logits, video_features, layer_results, layer_qs = self.decoder(kvs, m)

        if image_logits == None:
            task_logits = video_logits
        else:
            task_logits = [(vl + il) / 2 for vl, il in zip(video_logits, image_logits)]

        if with_video_features:
            features["video"] = video_features

        features["layer_results"] = layer_results

        features["layer_qs"] = layer_qs

        return task_logits, features

    def forward(self, x, y, m, comp=None, speed=None, train=False, single_task=None, *args, **kargs):
        device = x.device
        b, t, c, h, w = x.shape

        task_logits, features = self.predict(
            x,
            m,
            with_video_features=True,
            train=train
        )

        video_features = features["video"]

        task_losses = [
            loss_fn(logits, labels) if single_task == None or i == single_task else 0
            for i, loss_fn, logits, labels in zip(range(len(self.losses)), self.losses, task_logits, y)
        ]

        if not train:
            return task_losses, task_logits

        other_losses = {}

        layer_qs = features["layer_qs"]
        if not self.config.train_mode.query_guide.type == QueryGuideMode.NONE:
            # guide video learner class token
            if self.config.train_mode.query_guide.type == QueryGuideMode.NORMAL:
                other_losses["cls_sim"] = (
                    torch.sum(
                        (
                            1 - torch.nn.functional.cosine_similarity(
                                layer_qs.flatten(2, 3),
                                self.video_semantic_queries.to(layer_qs.device),
                                dim=-1
                            )
                        ) / 2,
                        dim=-1
                    )
                ).mean(1)

        return task_losses, task_logits, other_losses

    def configure_optimizers(self, lr):
        params = [i for i in self.parameters() if i.requires_grad]
        if (self.config.optimizer == Optimizers.SGD):
            return torch.optim.SGD(
                params=params,
                lr=lr,
                weight_decay=self.config.weight_decay,
                momentum=0.95
            )
        elif (self.config.optimizer == Optimizers.ADAMW):
            return torch.optim.AdamW(
                params=params,
                lr=lr,
                weight_decay=self.config.weight_decay
            )

    def _transform(self, n_px):
        if (self.config.foundation == Foundations.CLIP or self.config.foundation == Foundations.FARL):
            """
            Ported from:
            https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/clip.py#L79
            """
            return T.Compose([
                T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(n_px),
                T.ConvertImageDtype(torch.float32),
                T.Normalize((0.48145466, 0.4578275, 0.40821073),
                            (0.26862954, 0.26130258, 0.27577711)),
            ])
        elif (self.config.foundation == Foundations.DINO2):
            return T.Compose([
                T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(n_px),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)
                ),
            ])
        else:
            raise NotImplementedError()

    def train(self, mode=True):
        super().train(mode)
        if (mode):
            self.encoder.eval()
            self.encoder.frame_prompt_drop.train()
        return self


class Adaptor(nn.Module):
    def __init__(self, config, detector, num_frames=50):
        super().__init__()
        width = detector.encoder.width
        patches = (detector.encoder.input_resolution // detector.encoder.patch_size)**2
        self.layer_blocks = []
        self.residual = True
        for i in range(len(detector.layer_indices)):
            blk = {}
            for j in ["k", "v"]:
                _name = f"l{i}_{j}"

                if (config.adapter.struct.type == "768-x-768"):
                    inner_dim = int(config.adapter.struct.x)
                    setattr(
                        self,
                        _name,
                        torch.nn.Sequential(
                            torch.nn.Linear(width, inner_dim, bias=False),
                            torch.nn.GELU(),
                            torch.nn.LayerNorm(inner_dim),
                            torch.nn.Dropout(config.dropout / 5),

                            torch.nn.Linear(inner_dim, width, bias=False),
                            torch.nn.Dropout(config.dropout)
                        )
                    )
                elif (config.adapter.struct.type == "legacy-768-x-768"):
                    inner_dim = int(config.adapter.struct.x)
                    setattr(
                        self,
                        _name,
                        torch.nn.Sequential(
                            torch.nn.Linear(width, inner_dim, bias=False),
                            torch.nn.GELU(),
                            torch.nn.LayerNorm(inner_dim),
                            torch.nn.Linear(inner_dim, width, bias=False),
                            torch.nn.Dropout(config.dropout)
                        )
                    )
                elif (config.adapter.struct.type == "768-x-768-nln"):
                    inner_dim = int(config.adapter.struct.x)

                    setattr(
                        self,
                        _name,
                        torch.nn.Sequential(
                            torch.nn.Linear(width, inner_dim, bias=False),
                            torch.nn.LayerNorm((patches, inner_dim)),
                            torch.nn.GELU(),
                            torch.nn.Dropout(config.dropout / 10),

                            torch.nn.Linear(inner_dim, width, bias=False),
                            torch.nn.Dropout(config.dropout)
                        )
                    )
                elif (config.adapter.struct.type == "768-x-768-ln"):
                    inner_dim = int(config.adapter.struct.x)

                    setattr(
                        self,
                        _name,
                        torch.nn.Sequential(
                            torch.nn.Linear(width, inner_dim, bias=False),
                            torch.nn.LayerNorm(inner_dim),
                            torch.nn.GELU(),
                            torch.nn.Dropout(config.dropout / 10),

                            torch.nn.Linear(inner_dim, width, bias=False),
                            torch.nn.Dropout(config.dropout)
                        )
                    )
                elif (config.adapter.struct.type == "768-x-768-z0"):
                    inner_dim = int(config.adapter.struct.x)
                    # create module
                    module = torch.nn.Sequential(
                        torch.nn.Linear(width, inner_dim, bias=False),
                        torch.nn.LayerNorm(inner_dim),
                        torch.nn.GELU(),
                        torch.nn.Dropout(config.dropout / 10),
                        torch.nn.Linear(inner_dim, width, bias=False),
                        torch.nn.Dropout(config.dropout)
                    )

                    # initialize with zero weight
                    module[1].weight.data.zero_()
                    module[-2].weight.data.zero_()

                    # set attribute
                    setattr(
                        self,
                        _name,
                        module
                    )
                elif (config.adapter.struct.type == "768-bn"):
                    setattr(
                        self,
                        _name,
                        torch.nn.Sequential(
                            torch.nn.Linear(768, 768, bias=False),
                            torch.nn.BatchNorm2d(num_frames),
                            torch.nn.Dropout(config.dropout)
                        )
                    )
                elif (config.adapter.struct.type == "768-xxx-768"):
                    inner_dim = int(config.adapter.struct.x)
                    setattr(
                        self,
                        _name,
                        torch.nn.Sequential(
                            torch.nn.Linear(width, inner_dim, bias=False),
                            torch.nn.GELU(),
                            torch.nn.Dropout(config.dropout / 5),

                            torch.nn.Linear(inner_dim, inner_dim, bias=False),
                            torch.nn.GELU(),
                            torch.nn.Dropout(config.dropout / 5),

                            torch.nn.Linear(inner_dim, width, bias=False),
                            torch.nn.Dropout(config.dropout)
                        )
                    )
                elif (config.adapter.struct.type == "linear"):
                    self.residual = False
                    # create module
                    module = torch.nn.Sequential(
                        torch.nn.Linear(width, width, bias=False),
                        torch.nn.Dropout(config.dropout)
                    )

                    # initialize with zero weight
                    module[0].weight.data = torch.eye(width)

                    # set attribute
                    setattr(
                        self,
                        _name,
                        module
                    )
                else:
                    raise NotImplemented()

                blk[j] = getattr(self, _name)
            self.layer_blocks.append(blk)

    def forward(self, kvs):
        b, t, p, h, d = kvs[0]['k'].shape
        # perform compression invariant transformation, per layer per key has it's transform matrix
        for i in range(len(kvs)):
            for k in self.layer_blocks[i].keys():
                features = (self.layer_blocks[i][k](kvs[i][k].view((b, t, p, -1)))).view((b, t, p, h, d))
                if self.residual:
                    kvs[i][k] = kvs[i][k] + features
                else:
                    kvs[i][k] = features
        return kvs


class VPT(nn.Module):
    """
    Deepfake video detector (also a video classifier)

    A series of video frames are first fed into CLIP image encoder
    we export key and values for each frame in each layer of CLIP ViT
    these k, v then further flatten on the temporal dimension
    resulting in a series of tokens for each video.

    The decoder aggregates the exported key and values in an attention manner 
    a CLS query token is learned during decoding.
    """
    @staticmethod
    def get_default_config():
        C = CN()
        C.architecture = "ViT-B/16"
        C.name = "VPT"
        C.out_dim = []
        C.losses = []
        C.frame_prompts = 20
        C.prompt_mode = "deepc"
        # regularization
        C.dropout = 0.0
        C.weight_decay = 0.01
        # optimizer
        C.optimizer = Optimizers.SGD.name.lower()
        # attention
        C.attn_record = False
        # train mode
        C.train_mode = CN()
        # > query guide
        C.train_mode.query_guide = CN()
        C.train_mode.query_guide.type = QueryGuideMode.NONE.name.lower()
        # - for normal mode
        C.train_mode.query_guide.path = ""
        C.train_mode.query_guide.key = ""
        C.train_mode.query_guide.parts = []
        C.train_mode.query_guide.num_layers = -1
        return C

    @staticmethod
    def validate_config(config):
        config = config.clone()
        config.defrost()

        assert len(config.architecture) > 0

        assert type(config.frame_prompts) == int
        assert type(config.prompt_mode) == str

        assert type(config.out_dim) == list

        assert type(config.losses) == list

        assert type(config.dropout) == float
        assert 0 <= config.dropout <= 1

        assert type(config.weight_decay) == float
        assert 0 <= config.weight_decay <= 1

        config.optimizer = int(Optimizers[config.optimizer.upper()])

        assert type(config.attn_record) == bool

        config.train_mode.query_guide.type = int(QueryGuideMode[config.train_mode.query_guide.type.upper()])
        if config.train_mode.query_guide.type == QueryGuideMode.NORMAL:
            assert config.frame_prompts > 0
            assert len(config.train_mode.query_guide.path) > 0
            assert len(config.train_mode.query_guide.key) > 0
            assert len(config.train_mode.query_guide.parts) > 0
            assert config.train_mode.query_guide.num_layers > 0

        config.freeze()
        return config

    def __init__(self, config, num_frames, accelerator):
        super().__init__()
        self.config = self.validate_config(config)

        with accelerator.main_process_first():
            self.encoder = disable_gradients(
                clip.load(
                    config.architecture,
                    frame_prompts=config.frame_prompts,
                    prompt_mode=config.prompt_mode,
                    attn_record=config.attn_record
                )[0].visual.float()
            )
            if (self.encoder.prompts > 0):
                self.encoder.frame_prompt_embeddings.requires_grad_(True)

        self.out_dim = config.out_dim
        self.weight_decay = config.weight_decay
        self.optimizer = config.optimizer
        self.losses = []
        self.task_proj = []
        self.cls_drop = nn.Dropout(self.config.dropout)
        self.cls_ln = nn.LayerNorm(self.encoder.width)

        for loss in config.losses:
            if type(loss) == str:
                self.losses.append(globals()[loss]())
            else:
                self.losses.append(globals()[loss.name](**(dict(loss.args) if "args" in loss else {})))

        # class linear prob
        scale = self.encoder.width**-0.5
        for i, v in enumerate(self.out_dim):
            name = f"proj{i}_{v}"
            setattr(self, name, nn.Parameter(scale * torch.randn((self.encoder.width, v))))
            self.task_proj.append(getattr(self, name))

        # transformations
        self.transform = self._transform(self.encoder.input_resolution)

        # training requirements
        if (self.config.train_mode.query_guide.type == QueryGuideMode.NORMAL):
            file_path = self.config.train_mode.query_guide.path
            parts = self.config.train_mode.query_guide.parts
            key = self.config.train_mode.query_guide.key
            with open(file_path, "rb") as f:
                prefetch_queries = pickle.load(f)
                self.semantic_queries = []
                for i in range(self.encoder.layers):
                    layer_results = []
                    for part in parts:
                        layer_results.append(prefetch_queries[key][part][i])
                    self.semantic_queries.append(torch.stack(layer_results))
                self.semantic_queries = torch.stack(self.semantic_queries)
                self.semantic_queries.requires_grad_(False)

    def predict(self, x, train=False):
        b, t, c, h, w = x.shape

        # get key and value from each CLIP ViT layer
        kvs = self.encoder(x.flatten(0, 1), train=train, with_out=True, with_q=True, with_prompt=True)

        task_logits = [
            self.cls_ln(
                self.cls_drop(
                    kvs[-1]["out"][:, 0]
                )
            ) @ m
            for m in self.task_proj
        ]

        for i in range(len(task_logits)):
            logits_l2_distance = torch.norm(task_logits[i], dim=-1, keepdim=True)
            task_logits[i] = 5 * task_logits[i] / (logits_l2_distance + 1e-10)

        return task_logits, {"kvs": kvs}

    def forward(self, x, y, m, comp=None, speed=None, train=False, single_task=None, *args, **kargs):
        device = x.device
        b, t, c, h, w = x.shape

        task_logits, features = self.predict(x, train=train)

        task_losses = [
            loss_fn(logits, labels) if single_task == None or i == single_task else 0
            for i, loss_fn, logits, labels in zip(range(len(self.losses)), self.losses, task_logits, y)
        ]

        if not train:
            return task_losses, task_logits

        other_losses = {}

        # TODO: find the best setting for VPT.
        if not self.config.train_mode.query_guide.type == QueryGuideMode.NONE:
            if self.config.train_mode.query_guide.type == QueryGuideMode.NORMAL:
                # create stack of features for layer wise prompt tuning parameters.
                kvs = features["kvs"]
                layer_prompt_qs = torch.stack(
                    [kv["q"][:, 1:1 + self.encoder.prompts].flatten(-2) for kv in kvs],
                    dim=1
                )

                cls_part_layer = self.config.train_mode.query_guide.num_layers
                if (cls_part_layer > 0):
                    frame_prompts = self.encoder.prompts
                    num_queries = self.semantic_queries.shape[1]
                    repeat_spread_semantic_queries = self.semantic_queries[:cls_part_layer].repeat(
                        (1, frame_prompts // num_queries, 1)
                    )
                    if (frame_prompts % num_queries):
                        repeat_spread_semantic_queries = torch.cat(
                            (
                                repeat_spread_semantic_queries,
                                self.semantic_queries[:cls_part_layer, :(frame_prompts % num_queries)]
                            ),
                            dim=1
                        )
                    other_losses["cls_sim"] = (
                        (
                            1 - torch.nn.functional.cosine_similarity(
                                layer_prompt_qs[:, :cls_part_layer],
                                repeat_spread_semantic_queries.to(layer_prompt_qs.device),
                                dim=-1
                            )
                        ) / 2
                    )
        return task_losses, task_logits, other_losses

    def _transform(self, n_px):
        """
        Ported from:
        https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/clip.py#L79
        """
        return T.Compose([
            T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(n_px),
            T.ConvertImageDtype(torch.float32),
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711)),
        ])

    def configure_optimizers(self, lr):
        params = [i for i in self.parameters() if i.requires_grad]
        if (self.config.optimizer == Optimizers.SGD):
            return torch.optim.SGD(
                params=params,
                lr=lr,
                weight_decay=self.config.weight_decay,
                momentum=0.95
            )
        elif (self.config.optimizer == Optimizers.ADAMW):
            return torch.optim.AdamW(
                params=params,
                lr=lr,
                weight_decay=self.config.weight_decay
            )

    def train(self, mode=True):
        super().train(mode)
        if (mode):
            self.encoder.eval()
            if (self.encoder.prompts > 0):
                self.encoder.frame_prompt_drop.train()
        return self
