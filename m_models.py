import torch
import torch.nn as nn
import numpy as np
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoTokenizer
from cpe import CLIPParameterEfficient
import os



class MYCLIP(nn.Module):
    def __init__(self, args):
        super(MYCLIP, self).__init__()
        self.args = args

        self.clip_pe = CLIPParameterEfficient(
            L_g=2,                      
            deep_g=24,
            text_deep_replace_method="replace",
            vision_deep_replace_method="accumulate",
            training_phase=args.training_phase   
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.domain_stats = None

        for name, param in self.clip_pe.clip_model.named_parameters():
            param.requires_grad = False

        prompt_prefix = "This is a satellite image of "
        classnames = [name.replace("_", " ") for name in args.class_list]
        self.prompts = [prompt_prefix + name for name in classnames]

        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    @torch.no_grad()
    def frozen_image_features(self, images: torch.Tensor):
        """
        Extract L2‑normalised features from the *frozen* CLIP vision encoder,
        without any prompt tuning.
        """
        feats = self.clip_pe.clip_model.get_image_features(images)
        return feats / feats.norm(dim=-1, keepdim=True)

    def _mahalanobis_scores(self, feats: torch.Tensor):
        """
        feats: [B, D] extracted by frozen encoder
        Returns: scores [B, Q] where Q=len(self.domain_stats)
        """
        import math, torch
        B, D = feats.shape
        scores = []
        for (mu, inv, logdet) in self.domain_stats:
            mu = mu.to(feats.device)
            inv = inv.to(feats.device)
            delta = feats - mu                    # [B, D]
            maha = (delta @ inv * delta).sum(dim=-1)  # [B]
            score = -0.5 * (maha + logdet + D * math.log(2 * math.pi))
            scores.append(score)
        return torch.stack(scores, dim=-1)         # [B, Q]

    def _forward_fusion(self, images: torch.Tensor):
        """
        Inference forward pass with input‑specific fused prompts.
        Returns (image_features, text_features, logit_scale) as usual.
        """
        import torch
        device = images.device
        B = images.size(0)

        # 1) frozen feats → domain scores → softmax weights
        frozen_feat = self.frozen_image_features(images)     # [B, D]
        scores = self._mahalanobis_scores(frozen_feat)       # [B, Q]
        weights = scores.softmax(dim=-1)                     # [B, Q]

        # 2) vision branch with fused prompt
        fused_prompt_v = self.clip_pe.get_fused_g_values(weights)        # [B, deep_g, L_g, d_text]
        vision_prompt = self.clip_pe.prompt_proj(fused_prompt_v)         # → vision dim
        img_out = self.clip_pe.vision_model(images, vision_prompt)
        img_feat = self.clip_pe.image_proj(img_out)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)        # [B, D]

        # 3) text branch — use batch‑average weights to keep (C,D) shape
        w_mean = weights.mean(dim=0, keepdim=True)                       # [1, Q]
        fused_prompt_t = self.clip_pe.get_fused_g_values(w_mean)         # [1, deep_g, L_g, d_text]
        C = len(self.prompts)
        tok = self.tokenizer(self.prompts, padding=True, return_tensors="pt").to(device)
        text_prompt = fused_prompt_t.repeat(C, 1, 1, 1)                  # [C, deep_g, L_g, d_text]
        text_out = self.clip_pe.text_model(tok["input_ids"], tok["attention_mask"], text_prompt)
        txt_feat = self.clip_pe.text_proj(text_out)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)        # [C, D]

        return img_feat, txt_feat, self.logit_scale.exp()

    def forward(self, images: torch.Tensor):
        """
        Training phase: behave exactly as before.
        Inference phase (self.domain_stats not None): use dynamic prompt fusion.
        """
        if self.domain_stats is None:
            # ---------- Original training path ----------
            device = images.device
            B = images.size(0)
            current_g = self.clip_pe.get_current_g_values()
            vision_prompts = self.clip_pe.prompt_proj(current_g.unsqueeze(0).repeat(B, 1, 1, 1))
            img_out = self.clip_pe.vision_model(images, vision_prompts)
            img_feat = self.clip_pe.image_proj(img_out)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            tok = self.tokenizer(self.prompts, padding=True, return_tensors="pt").to(device)
            text_prompts = current_g.unsqueeze(0).repeat(tok["input_ids"].size(0), 1, 1, 1).to(device)
            txt_out = self.clip_pe.text_model(tok["input_ids"], tok["attention_mask"], text_prompts)
            txt_feat = self.clip_pe.text_proj(txt_out)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            return img_feat, txt_feat, self.logit_scale.exp()
        else:
            # ---------- Inference with prompt fusion ----------
            return self._forward_fusion(images)
