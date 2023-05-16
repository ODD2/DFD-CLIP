import random
from math import comb
from itertools import combinations
from collections import OrderedDict

from yacs.config import CfgNode as CN
import torch
from torch import nn
from . import clip
import torchvision.transforms as T

def mse(logits,y):
    value =  torch.pow(
        logits[:,:140].softmax(dim=-1) @ torch.tensor([ i for i in range(140) ]).float().to(logits.device) - y,
        2
    )
    return value / 1000


def kl_div(*args,**kargs):
    def driver(logits,y):
        return torch.nn.functional.kl_div(torch.nn.functional.log_softmax(logits,dim=1), y, reduction='none')
    return driver


def auc_roc(weight=None,*args,**kargs):
    def driver(logits,y,_weight=weight):
        if(_weight):
            _weight = torch.tensor(_weight,device=logits.device)
        return torch.nn.functional.cross_entropy(
            logits,
            y,
            weight=_weight,
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

    def __init__(self, embed_dim, n_head):
        def smax(q, k, m):
            """
            softmax in the original Transformer
            """
            aff = torch.einsum('nqhc,nkhc->nqkh', q / (q.size(-1) ** 0.5), k)
            aff = aff.masked_fill(~m, float('-inf'))
            aff = aff.softmax(dim=-2)
            return aff

        def coda(q, k, m):
            """
            Compositional De-Attention Networks, Neurips 19
            """
            norm = q.size(-1) ** 0.5
            aff = torch.einsum('nqhc,nkhc->nqkh', q / norm, k).tanh()
            gate = -(q - k).abs().sum(-1).unsqueeze(1) / norm
            gate = 2 * gate.sigmoid().masked_fill(~m, 0.)
            return aff * gate

        super().__init__()
        self.activations = [smax, coda]
        self.n_act = len(self.activations)  # softmax and coda
        self.in_proj = nn.Linear(embed_dim, self.n_act * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.embed_dim = embed_dim
        self.n_head = n_head

    def forward(self, q, k, v, m):
        qs = self.in_proj(q).view(*q.shape[:2], self.n_head, -1).split(self.embed_dim // self.n_head, -1)
        m = m.unsqueeze(1).unsqueeze(-1)

        aff = 0
        for i in range(self.n_act):
            aff += self.activations[i](qs[i], k, m) / self.n_act

        mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)

        return self.out_proj(mix.flatten(-2))


class ResidualAttentionBlock(nn.Module):
    """
    Modified from:
    https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L171
    """

    def __init__(self, d_model: int, n_head: int, config, reference_layer: nn.Module):
        super().__init__()

        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(config.dropout)),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

        self._apply_reference(reference_layer)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, m: torch.Tensor):
        return self.attn(q, k, v, m)

    def forward(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor, m: torch.Tensor):
        x = x + self.attention(self.ln_1(x), k, v, m)
        x = x + self.mlp(self.ln_2(x))
        return x

    def _apply_reference(self, ref: nn.Module):
        """
        use CLIP weights to initialize decoder
        """
        self.ln_1.load_state_dict(ref.ln_1.state_dict())
        self.mlp.load_state_dict(ref.mlp.state_dict())
        self.ln_2.load_state_dict(ref.ln_2.state_dict())


class Transformer(nn.Module):
    """
    Modified from:
    https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L195
    """

    def __init__(self, width: int, heads: int, config, reference_layers: nn.Sequential):
        super().__init__()
        self.width = width
        self.resblocks = []
        for layer in reference_layers:
            self.resblocks.append(ResidualAttentionBlock(width, heads, config, layer))

        self.resblocks = nn.Sequential(*self.resblocks)

    def forward(self, x: torch.Tensor, kvs, m):
        for blk, kv in zip(self.resblocks, kvs):
            x = blk(x, kv['k'], kv['v'], m)

        return x


class Decoder(nn.Module):
    """
    The decoder aggregates the keys and values exported from CLIP ViT layers
    and predict the truthfulness of a video clip

    The positional embeddings are shared across patches in the same spatial location
    """

    def __init__(self, detector, config, num_frames):
        super().__init__()
        width = detector.encoder.width
        heads = detector.encoder.heads
        output_dims = config.out_dim
        scale = width ** -0.5

        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_frames, 1, heads, width // heads))
        self.ln_pre = LayerNorm(width)
        self.drop_pre = torch.nn.Dropout(config.dropout)
        self.transformer = Transformer(
            width,
            heads,
            config,
            reference_layers=[
                detector.encoder.transformer.resblocks[i]
                for i in detector.layer_indices
            ],
        )
        self.ln_post = LayerNorm(width)
        self.drop_post = torch.nn.Dropout(config.dropout)
        self.projections  = []

        for i,output_dim in enumerate(output_dims):
            _name = f"proj{i}x{output_dim}"
            setattr(self,_name, nn.Parameter(scale * torch.randn(width, output_dim)))
            self.projections.append(getattr(self,_name))

    def forward(self, kvs, m):
        m = m.repeat_interleave(kvs[0]['k'].size(2), dim=-1)

        kvs = [
            {
                k: (v + self.positional_embedding).flatten(1, 2) for k, v in kv.items()
            }
            for kv in kvs
        ]
        
        x = self.class_embedding.view(1, 1, -1).repeat(kvs[0]['k'].size(0), 1, 1)
        x = self.ln_pre(self.drop_pre(x))
        x = self.transformer(x, kvs, m)
        x = self.ln_post(self.drop_post(x))
        video_feature = x.squeeze(1)
        task_logits = [video_feature @ projection for projection in self.projections]

        return task_logits, video_feature


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
        C.architecture = 'ViT-B/16'
        C.decode_mode = "stride"
        C.decode_stride = 2
        C.decode_indices = []
        C.out_dim = []
        C.losses = []
        # adapter configurations
        C.adapter = CN(new_allowed=True)
        C.adapter.type = "none"
        # training configurations
        C.train_mode = CN(new_allowed=True)
        C.train_mode.type = "normal"
        # regularization
        C.dropout = 0.0
        C.weight_decay = 0.01
        return C

    def __init__(self, config, num_frames, accelerator):
        super().__init__()
        assert config.decode_mode in ["stride","index"]

        with accelerator.main_process_first():
            self.encoder = disable_gradients(clip.load(config.architecture)[0].visual.float())

        self.decode_mode = config.decode_mode
        self.out_dim = config.out_dim
        self.weight_decay = config.weight_decay
        self.train_mode = config.train_mode
        self.losses = []
        
        
        for loss in config.losses:
            if type(loss) == str:
                self.losses.append(globals()[loss]())
            else:
                self.losses.append(globals()[loss.name](**(dict(loss.args) if "args" in loss else {})))


        if(self.decode_mode == "stride"):
            self.layer_indices = list(range(0,len(self.encoder.transformer.resblocks),config.decode_stride))
        elif(self.decode_mode == "index"):
            self.layer_indices = config.decode_indices
        else:
            raise Exception(f"Unknown decode type: {self.decode_type}")
        
        self.decoder = Decoder(self, config, num_frames)
        
        if(config.adapter.type == "none"):
            self.adapter = None
        else:
            self.adapter = CompInvAdapter(config,self,num_frames=num_frames)
            if(config.adapter.type == "pretrain"):
                data = torch.load(config.adapter.path)
                data = {
                    '.'.join(k.split('.')[1:]): v  for k,v in  data.items() if "adapter"  in k
                }
                self.adapter.load_state_dict(data)
                if(config.adapter.frozen):
                    self.adapter = disable_gradients(self.adapter)

        # transformations
        self.transform = self._transform(self.encoder.input_resolution)

        # trainable parameters
        if(self.train_mode.type == "temporal_ranking"):
            self.ranking_transform_param = nn.Parameter((self.encoder.width**-0.5) * torch.randn(self.encoder.width, 1),requires_grad=True)
        
    def predict(self, x, m, features=False):
        b, t, c, h, w = x.shape
        with torch.no_grad():
            # get key and value from each CLIP ViT layer
            kvs = self.encoder(x.flatten(0, 1))
            # discard original CLS token and restore temporal dimension
            kvs = [{k: v[:, 1:].unflatten(0, (b, t)) for k, v in kv.items()} for kv in kvs]

            kvs = [kvs[i] for i in self.layer_indices]

        if self.adapter:
            kvs = self.adapter(kvs)

        task_logits,video_features = self.decoder(kvs, m)

        for i in range(len(task_logits)):
            logits_l2_distance = torch.norm(task_logits[i],dim=-1,keepdim=True)
            task_logits[i] = 5 * task_logits[i] / (logits_l2_distance + 1e-10)

        if(not features):
            return task_logits
        else:
            return task_logits, video_features

    def forward(self, x, y, m, c=None, s=None, train=False,single_task=None,*args,**kargs):
        b = x.shape[0]
        task_logits, video_features = self.predict(x, m,features=True)
        
        task_losses = [
            loss_fn(logits, labels) if single_task==None or i == single_task else 0
            for i, loss_fn,logits,labels in zip(range(len(self.losses)),self.losses,task_logits,y)
        ]

        if not train:
            return task_losses, task_logits
        
        elif self.train_mode.type == "normal":
            return task_losses, task_logits
        
        elif self.train_mode.type == "temporal_ranking":
            speed_rank_index = torch.argsort(s,descending=True).tolist()
            speed_loss = torch.tensor(0.0,device=x.device)

            rank_logits = (video_features @ self.ranking_transform_param ).squeeze()
            rank_losses = []

            for rank in range(0,b-1):
                input1 = rank_logits[speed_rank_index[rank]].repeat(b-1-rank)
                input2 = rank_logits[speed_rank_index[rank+1:],...]
                target = torch.ones(b-1-rank,device=x.device)
                rank_losses.append(
                    torch.nn.functional.margin_ranking_loss(
                        input1,
                        input2,
                        target,
                        reduction="none"
                    )
                )

            speed_loss = torch.cat(rank_losses).mean()
            return task_losses, task_logits, 0.05*speed_loss
        
        elif self.train_mode.type == "temporal_triplet":
            speed_rank_index = torch.argsort(s,descending=True).tolist()
            speed_loss = torch.tensor(0.0,device=x.device)

            margin_rounds = min(comb(b,3),10)
            indices = list(range(b))
            random.shuffle(indices)
            _combinations = iter(combinations(indices,3))
            
            for _ in range(margin_rounds):
                b_index = next(_combinations)
                b_index = sorted(b_index,key=lambda _i:speed_rank_index.index(_i))
                speed_loss += torch.nn.functional.triplet_margin_loss(
                    anchor=video_features[b_index[0]],
                    positive=video_features[b_index[1]],
                    negative=video_features[b_index[2]],
                    margin=(s[b_index[2]]-s[b_index[1]])
                )
                speed_loss += torch.nn.functional.triplet_margin_loss(
                    anchor=video_features[b_index[2]],
                    positive=video_features[b_index[1]],
                    negative=video_features[b_index[0]],
                    margin=(s[b_index[1]]-s[b_index[0]])
                )

            return task_losses, task_logits, 0.01*speed_loss/margin_rounds/2
        else:
            raise NotImplementedError()
        
    def configure_optimizers(self, lr):
        return torch.optim.AdamW(
            params=[i for i in self.parameters() if i.requires_grad], 
            lr=lr,
            weight_decay=self.weight_decay
        )

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


class CompInvAdapter(nn.Module):
    def __init__(self,config,detector,num_frames=50):
        super().__init__()
        width = detector.encoder.width
        self.layer_blocks=[]
        for i in range(len(detector.layer_indices)):
            blk = {}
            for j in ["k","v"]:
                _name = f"l{i}_{j}"

                if(config.adapter.struct.type == "768-x-768"):
                    inner_dim = int(config.adapter.struct.x)
                    setattr(
                        self,
                        _name,
                        torch.nn.Sequential(
                            torch.nn.Linear(width,inner_dim,bias=False),
                            torch.nn.GELU(),
                            torch.nn.LayerNorm(inner_dim),
                            torch.nn.Linear(inner_dim,width,bias=False),
                            torch.nn.Dropout(config.dropout)
                        )
                    )
                elif(config.adapter.struct.type == "768-bn"):
                    setattr(
                        self,
                        _name,
                        torch.nn.Sequential(
                            torch.nn.Linear(768,768,bias=False),
                            torch.nn.BatchNorm2d(num_frames),
                            torch.nn.Dropout(config.dropout)
                        )
                    )
                else:
                    raise NotImplemented()
            
                blk[j] = getattr(self,_name)
            self.layer_blocks.append(blk)

    def forward(self, kvs):
        b,t,p,h,d = kvs[0]['k'].shape
        # perform compression invariant transformation, per layer per key has it's transform matrix
        kvs = [
            {
                k: (self.layer_blocks[i][k](v.view((b,t,p,-1)))).view((b,t,p,h,d)) + v for k, v in kv.items()
            }
            for i,kv in enumerate(kvs)
        ]

        return kvs


class CompInvEncoder(nn.Module):
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
        C.name = "CompInvEncoder"
        C.architecture = 'ViT-B/16'
        C.decode_mode = "stride"
        C.decode_stride = 2
        C.decode_indices = []
        C.adapter = CN(new_allowed=True)
        C.dropout = 0.0
        C.mode = 0
        return C

    def __init__(self, config, accelerator,num_frames=50,*args,**kargs):
        super().__init__()
        assert config.decode_mode in ["stride","index"]

        with accelerator.main_process_first():
            self.encoder = disable_gradients(clip.load(config.architecture)[0].visual.float())

        self.decode_mode = config.decode_mode
        
        if(self.decode_mode == "stride"):
            self.layer_indices = list(range(0,len(self.encoder.transformer.resblocks),config.decode_stride))
        elif(self.decode_mode == "index"):
            self.layer_indices = config.decode_indices
        else:
            raise Exception(f"Unknown decode type: {self.decode_type}")

        self.mode = int(config.mode)

        self.adapter = CompInvAdapter(config,self,num_frames=num_frames)
        self.transform = self._transform(self.encoder.input_resolution)
        
    def predict(self, x):
        b, t, c, h, w = x.shape
        with torch.no_grad():
            # get key and value from each CLIP ViT layer
            _kvs = self.encoder(x.flatten(0, 1))
            # discard original CLS token and restore temporal dimension
            _kvs = [{k: v[:, 1:].unflatten(0, (b, t)) for k, v in kv.items()} for kv in _kvs]
            _kvs = [_kvs[i] for i in self.layer_indices]

        kvs = self.adapter(_kvs)

        return kvs,_kvs

    def forward(self, x, comp,*args,**kargs):
        
        kvs,_kvs = self.predict(x)

        _l = len(self.layer_indices)

        _b,_t,_p,_h,_d =  kvs[0]['k'].shape 

        _w = _b//2

        device = kvs[0]['k'].device
        
        recon_loss = torch.tensor(0.0,device=device)
        match_loss = torch.tensor(0.0,device=device)

        recon_diff = torch.zeros((_t,_p,_h,_d),device=device)
        match_diff = torch.zeros((_t,_p,_h,_d),device=device)
        
        for i in range(_w):

            if(comp[i*2] == "raw"):
                raw_sup = i*2
                c23_sup = i*2+1
            else:
                raw_sup = i*2+1
                c23_sup = i*2

            for layer in range(_l):
                for subject in ["k","v"]:
                    if(self.mode == 0):
                        recon_diff += torch.abs(_kvs[layer][subject][raw_sup] - kvs[layer][subject][raw_sup])
                        match_diff += torch.abs(kvs[layer][subject][raw_sup] - kvs[layer][subject][c23_sup])
                    elif(self.mode == 1):
                        match_diff += torch.abs(_kvs[layer][subject][raw_sup] - kvs[layer][subject][c23_sup])
    
        recon_loss = torch.norm((recon_diff/(_w * _l *  2)).view((_p,_t,-1)).mean(dim=1))/_p
        match_loss = torch.norm((match_diff/(_w * _l *  2)).view((_p,_t,-1)).mean(dim=1))/_p

        return recon_loss, match_loss

    def configure_optimizers(self, lr):
        return torch.optim.AdamW(
            params=self.adapter.parameters(), 
            lr=lr
        )

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

