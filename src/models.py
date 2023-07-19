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
        self.in_proj = nn.Linear(embed_dim, self.n_act * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.embed_dim = embed_dim
        self.n_head = n_head

        self.aff = None
        self.qs = None

    def forward(self, q, k, v, m):
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

        return self.out_proj(mix.flatten(-2))


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

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, m: torch.Tensor):
        return self.attn(q, k, v, m)

    def forward(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor, m: torch.Tensor):
        x = x + self.attention(self.ln_1(x), k, v, m)
        x = x + self.mlp(self.ln_2(x))
        return x

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

        if (config.concat_ref):
            logging.debug("perform concatenation reference initialization.")
            current_layer = layer_indices[block_index]
            self.ln_1.load_state_dict(fetch_ln1_params(current_layer))
            self.ln_2.load_state_dict(fetch_ln2_params(current_layer))
            if (block_index < (len(layer_indices) - 1)):
                concat_layer = layer_indices[block_index + 1] - 1
                self.mlp.load_state_dict(fetch_mlp_params(concat_layer))
            else:
                self.mlp.load_state_dict(fetch_mlp_params(current_layer))
        else:
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

        # augmentive query

        if config.op_mode.aug_query:
            self.augment_queries = []
            for i in range(len(layer_indices) - 1):
                name = f"augment_query_{i}"
                setattr(self, name, nn.Parameter(torch.zeros(width)))
                self.augment_queries.append(getattr(self, name))

        self.resblocks = nn.Sequential(*self.resblocks)

    def forward(self, x: torch.Tensor, kvs, m):
        result = []

        for i, blk, kv in zip(range(len(self.resblocks)), self.resblocks, kvs):
            x = blk(x, kv['k'], kv['v'], m)
            result.append(x.unsqueeze(1))

            if self.config.op_mode.aug_query:
                logging.debug("perform augmentation query embedding.")
                if not (i == len(self.resblocks) - 1):
                    x = x + self.augment_queries[i]

        return torch.cat(result, dim=1)


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
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

        if self.config.op_mode.temporal_position:
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
        self.ln_post = LayerNorm(width)
        self.drop_post = torch.nn.Dropout(config.dropout)
        self.task_projections = []

        for i, output_dim in enumerate(output_dims):
            layer_matrices = []

            if self.config.op_mode.global_prediction:
                for l in detector.layer_indices:
                    _name = f"proj{i}x{output_dim}_L{l}"
                    setattr(self, _name, nn.Parameter(scale * torch.randn(width, output_dim)))
                    layer_matrices.append(getattr(self, _name))

            else:
                _name = f"proj{i}x{output_dim}"
                setattr(self, _name, nn.Parameter(scale * torch.randn(width, output_dim)))
                layer_matrices.append(getattr(self, _name))

            self.task_projections.append(layer_matrices)

    def forward(self, kvs, m):
        m = m.repeat_interleave(kvs[0]['k'].size(2), dim=-1)
        # add temporasitional embedding
        if self.config.op_mode.temporal_position:
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
        layer_results = self.transformer(x, kvs, m)

        x = layer_results.mean(dim=2)
        # drop the layer features if not required
        if (len(self.task_projections[0]) == 1):
            x = x[:, -1]

        x = self.drop_post(self.ln_post(x))
        video_feature = x.squeeze(1)

        if self.config.op_mode.global_prediction:
            logging.debug("perform weighted global prediction.")
            task_logits = [
                sum(
                    [
                        (
                            (video_feature[:, i] @ layer_matrices[i]) * (i + 1) /
                            ((1 + len(layer_matrices)) * len(layer_matrices) / 2)
                        )
                        for i in range(len(layer_matrices))
                    ]
                )
                for layer_matrices in self.task_projections
            ]
        else:
            logging.debug("perform last logit prediction.")
            task_logits = [
                video_feature @ layer_matrices[-1]
                for layer_matrices in self.task_projections
            ]

        return task_logits, video_feature


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

        # concatenation to reference layers
        C.concat_ref = False

        # adapter configurations
        C.adapter = CN()
        C.adapter.type = AdaptorMode.NONE.name.lower()
        C.adapter.struct = CN(new_allowed=True)
        C.adapter.frozen = False
        C.adapter.path = ""

        # training mode configurations
        C.train_mode = CN()
        # > temporal learning
        C.train_mode.temporal = TemporalMetric.NONE.name.lower()
        # > compression learning
        C.train_mode.compression = CompMetric.NONE.name.lower()
        # > patch mask
        C.train_mode.patch_mask = CN()
        C.train_mode.patch_mask.type = PatchMaskMode.NONE.name.lower()
        # - argument for GUIDED masking mode.
        C.train_mode.patch_mask.path = ""
        # - unmask patch ratio
        C.train_mode.patch_mask.ratio = 1.0
        # > nerf losses from raw label samples
        C.train_mode.nerf_raw = -1.0

        # operation mode configurations
        C.op_mode = CN()
        # > temporal positional embedding
        C.op_mode.temporal_position = True
        # > smax attention mode
        C.op_mode.attn_mode = AttnMode.PATCH.name.lower()
        # > attention activations
        C.op_mode.attn_driver = ["smax", "coda"]
        # > record attention scores
        C.op_mode.attn_record = False
        # > weighted global prediction
        C.op_mode.global_prediction = False
        # > augmentation queries
        C.op_mode.aug_query = False
        # > exponent moving average frames
        C.op_mode.ema_frame = -1.0

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

        config.decode_mode.type = int(DecodeMode[config.decode_mode.type.upper()])
        if config.decode_mode.type == DecodeMode.STRIDE:
            assert config.decode_mode.indices == int
            assert config.decode_mode.stride > 0, "invalid value for decode stride."
        elif config.decode_mode.type == DecodeMode.INDEX:
            assert type(config.decode_mode.indices) == list
            assert len(config.decode_mode.indices) > 0, "indices unspecified for decoding."

        assert type(config.out_dim) == list

        assert type(config.losses) == list

        assert type(config.concat_ref) == bool

        config.adapter.type = int(AdaptorMode[config.adapter.type.upper()])
        if (config.adapter.type == AdaptorMode.PRETRAIN):
            assert type(config.adapter.path) == str
            assert len(config.adapter.path) > 0
            assert type(config.adapter.frozen) == bool

        config.train_mode.temporal = int(TemporalMetric[config.train_mode.temporal.upper()])

        config.train_mode.compression = int(CompMetric[config.train_mode.compression.upper()])

        config.train_mode.patch_mask.type = int(PatchMaskMode[config.train_mode.patch_mask.type.upper()])
        assert type(config.train_mode.patch_mask.ratio) == float
        assert 0 < config.train_mode.patch_mask.ratio <= 1
        if config.train_mode.patch_mask.type == PatchMaskMode.GUIDE:
            assert type(config.train_mode.patch_mask.path) == str
            assert len(config.train_mode.patch_mask.path) > 0

        assert type(config.train_mode.nerf_raw) == float
        if config.train_mode.nerf_raw > 0:
            assert 0 <= config.train_mode.nerf_raw <= 1.0

        config.op_mode.attn_mode = [int(AttnMode[opt.upper()]) for opt in config.op_mode.attn_mode.split("+")]
        assert type(config.op_mode.attn_driver) == list
        assert type(config.op_mode.temporal_position) == bool
        assert type(config.op_mode.attn_record) == bool
        assert type(config.op_mode.global_prediction) == bool
        assert type(config.op_mode.aug_query) == bool

        assert type(config.op_mode.ema_frame) == float
        if config.op_mode.ema_frame > 0:
            assert 0 <= config.op_mode.ema_frame <= 1

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
                self.encoder = disable_gradients(clip.load(config.architecture)[0].visual.float())
            elif config.foundation == Foundations.DINO2:
                self.encoder = disable_gradients(DINOv2())
            elif config.foundation == Foundations.FARL:
                _model = clip.load("ViT-B/16")[0]
                _model.load_state_dict(
                    torch.load("./misc/FaRL-Base-Patch16-LAIONFace20M-ep64.pth")["state_dict"],
                    strict=False
                )
                self.encoder = _model.visual.float()
                self.encoder = disable_gradients(self.encoder)

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

        if (config.adapter.type == AdaptorMode.NONE):
            self.adapter = None
        elif (config.adapter.type == AdaptorMode.NORMAL):
            self.adapter = Adaptor(config, self, num_frames=num_frames)
            logging.info("Adapter operates without pretrained weights!!!")
        elif (config.adapter.type == AdaptorMode.PRETRAIN):
            self.adapter = Adaptor(config, self, num_frames=num_frames)
            data = torch.load(config.adapter.path)
            data = {
                '.'.join(k.split('.')[1:]): v for k, v in data.items() if "adapter" in k
            }
            self.adapter.load_state_dict(data)
            if (config.adapter.frozen):
                self.adapter = disable_gradients(self.adapter)
            logging.info(f"Adapter operates with pretrained weights:{config.adapter.path}")

        # transformations
        self.transform = self._transform(self.encoder.input_resolution)

        # trainable parameters
        if (self.config.train_mode.temporal == TemporalMetric.RANK):
            self.ranking_transform_param = nn.Parameter(
                (self.encoder.width**-0.5) * torch.randn(self.encoder.width, 1),
                requires_grad=True
            )

        if (self.config.train_mode.patch_mask.type == PatchMaskMode.GUIDE):
            with open(self.config.train_mode.patch_mask.path, "rb") as f:
                self.guide_map = pickle.load(f)

    def predict(self, x, m, with_video_features=False, with_adapt_features=False, train=False):
        b, t, c, h, w = x.shape

        with torch.no_grad():
            # get key and value from each CLIP ViT layer
            kvs = self.encoder(x.flatten(0, 1))
            # discard original CLS token and restore temporal dimension
            for i in range(len(kvs)):
                for k in kvs[i]:
                    kvs[i][k] = kvs[i][k][:, 1:].unflatten(0, (b, t))
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

        if self.adapter:
            logging.debug("perform feature adapting")
            kvs = self.adapter(kvs)

        task_logits, video_features = self.decoder(kvs, m)

        for i in range(len(task_logits)):
            logits_l2_distance = torch.norm(task_logits[i], dim=-1, keepdim=True)
            task_logits[i] = 5 * task_logits[i] / (logits_l2_distance + 1e-10)

        features = {}

        if with_video_features:
            features["video"] = video_features

        if with_adapt_features:
            if self.adapter:
                features["adapt"] = kvs
            else:
                raise Exception("cannot return adaptive features without an adapter")

        return task_logits, features

    def forward(self, x, y, m, comp=None, speed=None, train=False, single_task=None, *args, **kargs):
        device = x.device
        b, t, c, h, w = x.shape

        if (self.config.op_mode.ema_frame > 0):
            logging.debug("perform ema frame conversion")
            ema_ratio = self.config.op_mode.ema_frame
            _x = torch.zeros((b, 1, c, h, w), device=x.device)
            for i in range(t):
                _x = _x * ema_ratio + x[:, i].unsqueeze(1) * (1 - ema_ratio)
            x = _x
            m = m[:, 0].unsqueeze(1)

        task_logits, features = self.predict(
            x,
            m,
            with_video_features=True,
            with_adapt_features=(
                (not self.adapter == None) and
                (not self.config.train_mode.compression == CompMetric.NONE)
            ),
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

        # compression related losses
        if not self.config.train_mode.compression == CompMetric.NONE:
            logging.debug("enter compression block.")

            kvs = features["adapt"]
            _l = len(self.layer_indices)
            _b, _t, _p, _h, _d = kvs[0]['k'].shape
            _w = _b // 2

            recon_loss = torch.tensor(0.0, device=device)
            match_loss = torch.tensor(0.0, device=device)

            for i in range(_w):
                if (comp[i * 2] == "raw"):
                    raw_i = i * 2
                    c23_i = i * 2 + 1
                else:
                    raw_i = i * 2 + 1
                    c23_i = i * 2

                if self.config.train_mode.compression == CompMetric.FEATURE:
                    logging.debug("compute compression metric in feature mode.")
                    match_loss += torch.nn.functional.kl_div(
                        torch.log_softmax(video_features[c23_i], dim=-1),
                        torch.log_softmax(video_features[raw_i], dim=-1),
                        log_target=True
                    ) / (_w)
                elif self.config.train_mode.compression == CompMetric.SYNC:
                    logging.debug("compute compression metric in sync mode.")
                    for layer in range(_l):
                        for subject in ["k", "v"]:
                            match_loss += torch.nn.functional.kl_div(
                                torch.log_softmax(kvs[layer][subject][c23_i], dim=-1),
                                torch.log_softmax(kvs[layer][subject][raw_i], dim=-1),
                                log_target=True
                            ) / (_w * _l * 2)

            other_losses["recon"] = recon_loss
            other_losses["match"] = 100 * match_loss

        # nerf the strength of raw samples
        if self.config.train_mode.nerf_raw > 0:
            logging.debug("perform raw sample label nerfing.")
            nerf_power = min(self.config.train_mode.nerf_raw, 0)
            for i in range(len(task_losses)):
                for j in range(_b):
                    if comp[j] == "raw":
                        task_losses[i][j] *= nerf_power
                    else:
                        task_losses[i][j] *= (2 - nerf_power)

        # temporal related losses
        if not self.config.train_mode.temporal == TemporalMetric.NONE:
            logging.debug("enter temporal metric block.")
            speed_rank_index = torch.argsort(speed, descending=True).tolist()
            speed_loss = torch.tensor(0.0, device=x.device)

            if self.config.train_mode.temporal == TemporalMetric.RANK:
                logging.debug("perform temporal metric ranking.")
                rank_logits = (video_features @ self.ranking_transform_param).squeeze()
                rank_losses = []

                for rank in range(0, b - 1):
                    input1 = rank_logits[speed_rank_index[rank]].repeat(b - 1 - rank)
                    input2 = rank_logits[speed_rank_index[rank + 1:], ...]
                    target = torch.ones(b - 1 - rank, device=x.device)
                    rank_losses.append(
                        torch.nn.functional.margin_ranking_loss(
                            input1,
                            input2,
                            target,
                            reduction="none"
                        )
                    )

                speed_loss = torch.cat(rank_losses).mean()
                speed_loss = 0.05 * speed_loss

                other_losses["speed/rank"] = speed_loss

            elif self.config.train_mode.temporal == TemporalMetric.TRIPLET:
                logging.debug("perform temporal triplet metric learning.")
                margin_rounds = min(comb(b, 3), 10)

                indices = list(range(b))
                random.shuffle(indices)

                _combinations = iter(combinations(indices, 3))

                for _ in range(margin_rounds):
                    b_index = next(_combinations)
                    b_index = sorted(b_index, key=lambda _i: speed_rank_index.index(_i))

                    speed_loss += torch.nn.functional.triplet_margin_loss(
                        anchor=video_features[b_index[0]],
                        positive=video_features[b_index[1]],
                        negative=video_features[b_index[2]],
                        margin=torch.abs(speed[b_index[2]] - speed[b_index[1]])
                    )
                    speed_loss += torch.nn.functional.triplet_margin_loss(
                        anchor=video_features[b_index[2]],
                        positive=video_features[b_index[1]],
                        negative=video_features[b_index[0]],
                        margin=torch.abs(speed[b_index[1]] - speed[b_index[0]])
                    )

                speed_loss = 0.01 * speed_loss / (margin_rounds * 2)

                other_losses["speed/triplet"] = speed_loss

            else:
                raise NotImplementedError()

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
            for k in kvs[i].keys():
                features = (self.layer_blocks[i][k](kvs[i][k].view((b, t, p, -1)))).view((b, t, p, h, d))
                if self.residual:
                    kvs[i][k] = kvs[i][k] + features
                else:
                    kvs[i][k] = features
        return kvs
