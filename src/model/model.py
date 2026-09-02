import sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import hf_hub_download
from timm.data import resolve_data_config, create_transform

from model.backbone import ViTBackbone
from model.utils import TCAttentionExtract

class CrossAttnViT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self._device = torch.device("cpu")
        self.attention_extractor = None
        self._gate_params = {}
        self._gate_factor = 1.0

        #### Load Vision Model ####
        self.vision_model = ViTBackbone(config["vision_encoder"])
        assert self.image_size[0] % self.patch_size == 0, "Image resolution must be divisible by patch size"
        self.num_img_tokens = (self.image_size[0] // self.patch_size) ** 2 + self.vision_model.trunk.num_prefix_tokens
        self.feature_aggregation = config["vision_encoder"]["feature_aggregation"] # "cls" or "mean"
        assert self.feature_aggregation in ["cls", "mean"], "Feature aggregation must be either 'cls' or 'mean'"
        if self.feature_aggregation == "cls":
            assert self.vision_model.trunk.num_prefix_tokens > 0, "Model must have a cls token for cls feature aggregation"
        
        self.visual_dim = self.vision_model.trunk.embed_dim
        
        #### Load Language Model ####
        text_encoder = config['text_encoder']
        self.text_encoder_type = text_encoder
        self._word_vocab = None
        self._word_embedding = None

        if text_encoder == "learned":
            # Shallow trainable word embedding (no pretrained text model)
            import re
            from collections import Counter
            self.tokenizer = None
            self.text_model = None
            self.connector = None
            self.text_dim = self.visual_dim  # embed directly to visual dim
        elif "roberta" in text_encoder.lower():
            from transformers import RobertaTokenizer, RobertaModel
            self.tokenizer = RobertaTokenizer.from_pretrained(text_encoder)
            self.text_model = RobertaModel.from_pretrained(text_encoder).eval()
            self.text_dim = self.text_model.config.hidden_size
            for p in self.text_model.parameters():
                p.requires_grad = False
            #### Load Language-Image Connector ####
            self.connector = Connector(self.text_dim, self.visual_dim)
        else:
            raise NotImplementedError(f"Text encoder {text_encoder} currently not implemented")

        #### Mirror variant: GCA inside the text encoder (text = query, ViT patches = key/value) ####
        self._mirror_kv = None
        text_ca_layers = config.get("text_cross_attn_layers")
        if text_ca_layers:
            from model.crossattention import GatedCrossAttention
            self.text_gca = nn.ModuleDict({
                str(i): GatedCrossAttention(
                    i, dim=self.text_dim, ff_mult=2, use_ffn=False,
                    head_dim=64, num_heads=16,
                    use_gate=config["vision_encoder"].get("use_gate", True),
                    kv_dim=self.visual_dim,
                ) for i in text_ca_layers
            })
            for i in text_ca_layers:
                self.text_model.encoder.layer[i].register_forward_pre_hook(
                    self._make_text_gca_hook(str(i)), with_kwargs=True)
        else:
            self.text_gca = None

        #### Load Segmentation Head ####
        self.lin_seg_head = nn.Linear(self.visual_dim, 1, bias = True)
        nn.init.constant_(self.lin_seg_head.weight, 0)
        nn.init.constant_(self.lin_seg_head.bias, 0)

    def to(self, device):
        self.vision_model = self.vision_model.to(device)
        self.lin_seg_head = self.lin_seg_head.to(device)
        if self.text_model is not None:
            self.text_model = self.text_model.to(device)
        if self.connector is not None:
            self.connector = self.connector.to(device)
        if self._word_embedding is not None:
            self._word_embedding = self._word_embedding.to(device)
        if self.text_gca is not None:
            self.text_gca = self.text_gca.to(device)
        self._device = device
        return self

    def _make_text_gca_hook(self, key):
        """Forward pre-hook on a RoBERTa layer: apply text_gca[key] to its hidden_states input."""
        def hook(module, args, kwargs):
            if self._mirror_kv is None:
                return None
            if "hidden_states" in kwargs:
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = self.text_gca[key](kwargs["hidden_states"], self._mirror_kv, attn_mask=None)
            else:
                args = (self.text_gca[key](args[0], self._mirror_kv, attn_mask=None),) + tuple(args[1:])
            return args, kwargs
        return hook
    
    @classmethod
    def from_pretrained(cls, checkpoint_name, device=None):
        if os.path.isfile(checkpoint_name):
            ckpt_path = checkpoint_name
        else:
            ckpt_path = hf_hub_download(
                repo_id="JonaRuthardt/SteerViT",
                filename=checkpoint_name,
            )

        checkpoint = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )

        model = cls(checkpoint["config"])
        model.load_state_dict(
            checkpoint["state_dict"],
            strict=False,
        )
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        if device is not None:
            model = model.to(device)
        return model

    @classmethod
    def from_config(cls, backbone_name, device=None, cross_attn_layers=None,
                    resolution=336, feature_aggregation=None, pretrained=True,
                    condition_type="gca", use_gate=True,
                    text_encoder="roberta-large", text_cross_attn_layers=None):
        """Initialize from backbone name (no pretrained GCA).

        Args:
            backbone_name: timm model name.
            device: target device.
            cross_attn_layers: list of layer indices for GCA.
                               Default: every other layer [1,3,5,...].
            resolution: input image resolution.
            feature_aggregation: "cls" or "mean". Auto-detected if None
                                 (mean for models without CLS token).
        """
        import timm as _timm
        tmp = _timm.create_model(backbone_name, pretrained=False)
        num_blocks = len(tmp.blocks)
        has_cls = tmp.num_prefix_tokens > 0
        del tmp

        if cross_attn_layers is None:
            cross_attn_layers = list(range(1, num_blocks, 2))

        if feature_aggregation is None:
            feature_aggregation = "cls" if has_cls else "mean"

        config = {
            "vision_encoder": {
                "model_name": backbone_name,
                "resolution": resolution,
                "feature_aggregation": feature_aggregation,
                "cross_attn_layers": cross_attn_layers,
                "use_ffn": False,
                "cross_attn_ffn_mult": 2,
                "pretrained": pretrained,
                "condition_type": condition_type,
                "use_gate": use_gate,
            },
            "text_encoder": text_encoder,
        }
        if text_cross_attn_layers is not None:
            config["text_cross_attn_layers"] = list(text_cross_attn_layers)
        model = cls(config)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        if device is not None:
            model = model.to(device)
        return model

    @property
    def patch_size(self):
        return self.vision_model.trunk.patch_embed.patch_size[0]
    
    @property
    def feature_dim(self):
        return self.vision_model.trunk.embed_dim
    
    @property
    def image_size(self):
        return (self.vision_model.resolution, self.vision_model.resolution)
    
    def get_transforms(self):
        from torchvision import transforms
        vision_config = resolve_data_config({}, model=self.vision_model.trunk)
        size = self.image_size[0]
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=vision_config["mean"], std=vision_config["std"]),
        ])
        return transform

    def _init_word_embedding(self, data_root=None):
        """Lazily initialize learned word embedding from CLEVR vocab."""
        import re
        from collections import Counter

        if data_root is None:
            data_root = "/home/jungchun/data/clevr/CLEVR_v1.0"

        vocab = {"<pad>": 0, "<unk>": 1}
        q_path = os.path.join(data_root, "questions", "CLEVR_train_questions.json")
        import json
        with open(q_path) as f:
            q_data = json.load(f)
        for q in q_data["questions"]:
            words = re.findall(r"[a-z0-9]+", q["question"].lower())
            for w in words:
                if w not in vocab:
                    vocab[w] = len(vocab)

        self._word_vocab = vocab
        self._word_embedding = nn.Embedding(len(vocab), self.visual_dim,
                                            padding_idx=0).to(self._device)

    def _tokenize_words(self, texts: list[str]):
        """Tokenize texts to word IDs with padding."""
        import re
        if self._word_vocab is None:
            self._init_word_embedding()

        pad_id = self._word_vocab["<pad>"]
        unk_id = self._word_vocab["<unk>"]
        all_ids = []
        for t in texts:
            words = re.findall(r"[a-z0-9]+", t.lower())
            ids = [self._word_vocab.get(w, unk_id) for w in words]
            all_ids.append(ids)

        max_len = max(len(ids) for ids in all_ids)
        padded = torch.full((len(texts), max_len), pad_id, dtype=torch.long,
                            device=self._device)
        mask = torch.zeros(len(texts), max_len, dtype=torch.bool,
                           device=self._device)
        for i, ids in enumerate(all_ids):
            padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask[i, :len(ids)] = True
        return padded, mask

    def encode_text(self, texts: list[str]):
        """Encode text to conditioning space.

        Returns:
            text_feats: (B, T, visual_dim) projected text features for GCA.
            attn_mask: (B, num_img_tokens + T) attention mask with image prefix.
            raw_text_feats: (B, T, text_dim) raw text outputs.
        """
        if self.text_encoder_type == "learned":
            # Shallow trainable word embedding
            token_ids, word_mask = self._tokenize_words(texts)
            text_feats = self._word_embedding(token_ids)  # (B, T, visual_dim)
            attn_mask = torch.cat(
                (torch.ones(text_feats.size(0), self.num_img_tokens,
                            dtype=torch.bool, device=word_mask.device),
                 word_mask), dim=-1)
            return text_feats, attn_mask, text_feats

        # RoBERTa path
        roberta_dict = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        roberta_dict = {k: v.to(self.text_model.device) for k, v in roberta_dict.items()}
        raw_text_feats = self.text_model(**roberta_dict).last_hidden_state
        attn_mask = roberta_dict['attention_mask'].bool()

        text_feats = F.normalize(raw_text_feats, dim=-1)
        text_feats = self.connector(text_feats)

        attn_mask = torch.cat(
            (torch.ones(text_feats.size(0), self.num_img_tokens, dtype=torch.bool, device=attn_mask.device), attn_mask),
            dim=-1,
        )
        return text_feats, attn_mask, raw_text_feats

    def encode_text_mirror(self, texts: list[str], patch_kv: torch.Tensor):
        """Mirror variant: run RoBERTa with text_gca hooks attending to ViT patches.

        Args:
            texts: list of B strings.
            patch_kv: (B, P, visual_dim) ViT patch tokens used as GCA key/value.

        Returns:
            mem: (B, T, visual_dim) connector(L2-normalised last hidden state).
            attn_mask: (B, T) bool, True = real token.
        """
        roberta_dict = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        roberta_dict = {k: v.to(self.text_model.device) for k, v in roberta_dict.items()}
        self._mirror_kv = patch_kv
        try:
            raw_text_feats = self.text_model(**roberta_dict).last_hidden_state
        finally:
            self._mirror_kv = None
        attn_mask = roberta_dict['attention_mask'].bool()
        return self.connector(F.normalize(raw_text_feats, dim=-1)), attn_mask

    def encode_text_from_tokens(self, input_ids: 'torch.Tensor', attention_mask: 'torch.Tensor'):
        """Encode text from pre-tokenized inputs (skip tokenizer).

        Args:
            input_ids: (B, T) pre-tokenized input IDs.
            attention_mask: (B, T) attention mask.

        Returns:
            Same as encode_text(): (text_feats, attn_mask, raw_text_feats)
        """
        input_ids = input_ids.to(self.text_model.device)
        attention_mask = attention_mask.to(self.text_model.device)
        with torch.no_grad():
            raw_text_feats = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        attn_mask_bool = attention_mask.bool()

        text_feats = F.normalize(raw_text_feats, dim=-1)
        text_feats = self.connector(text_feats)

        attn_mask = torch.cat(
            (torch.ones(text_feats.size(0), self.num_img_tokens, dtype=torch.bool, device=attn_mask_bool.device), attn_mask_bool),
            dim=-1,
        )
        return text_feats, attn_mask, raw_text_feats

    def forward_with_conditioning(self, images: torch.Tensor, text_feats: torch.Tensor, attn_mask: torch.Tensor):
        """Forward pass with pre-computed text conditioning.

        Use this when you already have projected text features (e.g., for recurrent passes
        where visual pseudo-tokens are concatenated to text features).

        Args:
            images: (B, 3, H, W) input images.
            text_feats: (B, T, visual_dim) projected text features for GCA.
            attn_mask: (B, num_img_tokens + T) attention mask.

        Returns:
            img_feats: (B, N, visual_dim) output visual features.
        """
        return self.vision_model(images, text_feats, attn_mask=attn_mask)

    def forward(self, images: torch.Tensor, texts: torch.Tensor = None):
        if texts is not None:
            # Text conditioning
            assert images.size(0) == len(texts), "Batch size of images and texts must match"
            text_feats, attn_mask, _ = self.encode_text(texts)
        else:
            # Equivalent to vanilla base ViT model
            text_feats = attn_mask = None

        img_feats = self.vision_model(images, text_feats, attn_mask = attn_mask) #text conditioned img feats

        return img_feats
    
    @torch.no_grad()
    def get_dense_features(self, images: torch.Tensor, texts: list[str] = None):
        return self.forward(images.to(self._device), texts)[:, self.vision_model.trunk.num_prefix_tokens:, :]

    @torch.no_grad()
    def get_global_features(self, images: torch.Tensor, texts: list[str] = None):
        feats = self.forward(images.to(self._device), texts)
        if self.feature_aggregation == 'cls':
            return feats[:, 0, :]
        elif self.feature_aggregation == 'mean':
            return torch.mean(feats[:, self.vision_model.trunk.num_prefix_tokens:, :], dim=1)
        else:
            raise NotImplementedError(f"Feature aggregation {self.feature_aggregation} not implemented")

    @torch.no_grad()
    def get_heatmaps(self, images: torch.Tensor, texts: list[str] = None):
        img_feats = self.forward(images.to(self._device), texts)[:, self.vision_model.trunk.num_prefix_tokens:, :]
        heatmap_logits = self.lin_seg_head(img_feats).squeeze(-1)
        heatmaps = F.softmax(heatmap_logits, dim=1).view(images.size(0), 1, self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size)
        heatmaps = F.interpolate(heatmaps, size=self.image_size, mode='bilinear', align_corners=False)
        return heatmaps

    @torch.no_grad()
    def get_attention_heatmaps(self, images: torch.Tensor, texts: list[str] = None, **kwargs):
        if self.attention_extractor is None:
            self.attention_extractor = TCAttentionExtract(
                model=self,
                mode='eval',
                method='hook',
            )
        
        heatmaps = self.attention_extractor.get_attention_heatmaps(
            imgs=images.to(self._device), texts=texts, num_prefix_tokens=self.vision_model.trunk.num_prefix_tokens, **kwargs)
        
        return heatmaps
    
    def set_gate_factor(self, factor: float):
        for blk_idx, blk in enumerate(self.vision_model.trunk.blocks):
            gca = getattr(blk, "gated_cross_attn", None)
            if gca is not None:
                with torch.no_grad():
                    if blk_idx not in self._gate_params:
                        # Store original gate parameters
                        self._gate_params[blk_idx] = {
                            "attn_gate": gca.attn_gate.clone(),
                            "ff_gate": gca.ff_gate.clone() if hasattr(gca, "ff_gate") else None,
                        }
                    gca.attn_gate.copy_(self._gate_params[blk_idx]["attn_gate"] * float(factor))
                    if hasattr(gca, "ff_gate"):
                        gca.ff_gate.copy_(self._gate_params[blk_idx]["ff_gate"] * float(factor))
        self._gate_factor = factor
    
class Connector(nn.Module):
    def __init__(self, input_dim, output_dim=1152):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim, bias=True)
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        return self.mlp(x)