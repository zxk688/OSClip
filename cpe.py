import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPModel, AutoTokenizer

#############################################
### CLIP Text Encoder Parameter-Efficient ###
#############################################

class CLIPTextModelForPromptTuning(nn.Module):
    def __init__(self, model: object, deep_g: int, deep_replace_method: str = "replace"):
        '''
        CLIP Text Encoder for PE
        model: CLIP Text Encoder
        deep_g: number of layers to append prompts
        deep_replace_method: "replace", "accumulate", or "accumulate_same"
        '''
        super().__init__()
        self.model = model
        self.d_model = 768
        self.deep_g = deep_g
        self.deep_replace_method = deep_replace_method

    def forward(self, text_tokens: torch.Tensor, attn_mask: torch.Tensor, g_prompt: torch.Tensor):
        '''
        text_tokens: [batch_size, n_tokens]
        attn_mask: [batch_size, n_tokens]
        g_prompt: [batch_size, n_deep, n_prompts, d_model]
        '''
        bs = g_prompt.size(0)
        g_prompt = g_prompt.to(text_tokens.device)
        
        x = self.model.embeddings.token_embedding(text_tokens)
        g = g_prompt[:, 0]
        L_g = g.size(1)
        x = torch.cat([x[:,0:1,:], g, x[:,1:,:]], dim=1)
        x = x + self.model.embeddings.position_embedding(
            torch.arange(x.size(1), device=attn_mask.device).unsqueeze(0)
        )

        for i, l in enumerate(self.model.encoder.layers):
            if i > 0:
                if i < self.deep_g:
                    if self.deep_replace_method == "replace":
                        g = g_prompt[:, i]
                    elif self.deep_replace_method == "accumulate":
                        previous_g_out = x[:, 1:(L_g+1), :]
                        g = torch.cat([previous_g_out, g_prompt[:, i]], dim=1)
                    elif self.deep_replace_method == "accumulate_same":
                        g = torch.cat([g, g_prompt[:, i]], dim=1)
                    x = torch.cat([x[:, 0:1, :], g, x[:, (L_g+1):, :]], dim=1)
                    L_g = g.size(1)

            attn_mask_ = torch.cat([torch.ones(bs, L_g, device=attn_mask.device), attn_mask], dim=-1)
            res = x
            x = l.layer_norm1(x)

            q = l.self_attn.q_proj(x) * 0.125
            k = l.self_attn.k_proj(x)
            v = l.self_attn.v_proj(x)

            extended_attn_mask = (attn_mask_.unsqueeze(1).unsqueeze(1) == 0).float()
            extended_attn_mask[extended_attn_mask==1] = torch.finfo(x.dtype).min
            
            num_attention_heads = 12
            q = q.view(x.size(0), x.size(1), num_attention_heads, -1).transpose(1, 2)
            k = k.view(x.size(0), x.size(1), num_attention_heads, -1).transpose(1, 2)
            v = v.view(x.size(0), x.size(1), num_attention_heads, -1).transpose(1, 2)
            
            w = q @ k.transpose(-1, -2)
            w = w + extended_attn_mask
            w = w.softmax(dim=-1)
            v = (w @ v).transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
            x = l.self_attn.out_proj(v)
            x = res + x

            res = x
            x = l.layer_norm2(x)
            x = l.mlp(x)
            x = res + x

        x = self.model.final_layer_norm(x)
        index = text_tokens.argmax(dim=-1) + L_g
        return x[torch.arange(x.size(0)), index]


###############################################
### CLIP Vision Encoder Parameter-Efficient ###
###############################################

class CLIPVisionModelForPromptTuning(nn.Module):
    def __init__(self, model: object, deep_g: int, deep_replace_method: str = "accumulate"):
        '''
        CLIP Vision Encoder for PE
        model: CLIP Vision Encoder
        deep_g: number of layers to append prompts
        deep_replace_method: "replace", "accumulate", or "accumulate_same"
        '''
        super().__init__()
        self.model = model
        self.d_model = 1024
        self.deep_g = deep_g
        self.deep_replace_method = deep_replace_method

    def forward(self, image: torch.Tensor, g_prompt: torch.Tensor):
        '''
        image: [batch_size, 3, 224, 224]
        g_prompt: [batch_size, n_deep, n_prompts, d_model]
        '''
        x = self.model.embeddings(image)
        g = g_prompt[:, 0]
        x = torch.cat([x, g], dim=1)
        x = self.model.pre_layrnorm(x)
        L_g = g.size(1)
        
        for i, l in enumerate(self.model.encoder.layers):
            if i > 0:
                if i < self.deep_g:
                    if self.deep_replace_method == "replace":
                        g = g_prompt[:, i]
                    elif self.deep_replace_method == "accumulate":
                        previous_g_out = x[:, -L_g:, :]
                        g = torch.cat([previous_g_out, g_prompt[:, i]], dim=1)
                    elif self.deep_replace_method == "accumulate_same":
                        g = torch.cat([g, g_prompt[:, i]], dim=1)
                    x = torch.cat([x[:, :-L_g, :], g], dim=1)
                    L_g = g.size(1)

            res = x
            x = l.layer_norm1(x)

            q = l.self_attn.q_proj(x) * 0.125
            k = l.self_attn.k_proj(x)
            v = l.self_attn.v_proj(x)

            num_attention_heads = 16
            q = q.view(x.size(0), x.size(1), num_attention_heads, -1).transpose(1, 2)
            k = k.view(x.size(0), x.size(1), num_attention_heads, -1).transpose(1, 2)
            v = v.view(x.size(0), x.size(1), num_attention_heads, -1).transpose(1, 2)
            w = q @ k.transpose(-1, -2)
            w = w.softmax(dim=-1)
            v = (w @ v).transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
            x = l.self_attn.out_proj(v)
            x = res + x

            res = x
            x = l.layer_norm2(x)
            x = l.mlp(x)
            x = res + x

        return self.model.post_layernorm(x[:, 0, :])


################################
### CLIP Parameter-Efficient ###
################################

class CLIPParameterEfficient(nn.Module):
    def __init__(self, 
                 L_g: int = 2, 
                 deep_g: int = 24, 
                 text_deep_replace_method: str = "replace",
                 vision_deep_replace_method: str = "accumulate",
                 training_phase: str = "phase1"):
        '''
        CLIP Parameter-Efficient with domain-adaptive prompt tuning
        L_g: number of prompt tokens per layer
        deep_g: number of layers to attach prompts
        training_phase: "phase1" or "phase2"
        '''
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

        self.text_model = CLIPTextModelForPromptTuning(
            model=self.clip_model.text_model,
            deep_g=deep_g,
            deep_replace_method=text_deep_replace_method
        )
        self.vision_model = CLIPVisionModelForPromptTuning(
            model=self.clip_model.vision_model,
            deep_g=deep_g,
            deep_replace_method=vision_deep_replace_method
        )
        self.image_proj = self.clip_model.visual_projection
        self.text_proj = self.clip_model.text_projection
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

        self.prompt_proj = nn.Linear(self.text_model.d_model, self.vision_model.d_model)
        
        self.g_values_target1 = nn.Parameter(torch.zeros(deep_g, L_g, self.text_model.d_model))
        self.g_values_target2 = nn.Parameter(torch.zeros(deep_g, L_g, self.text_model.d_model))
        nn.init.xavier_uniform_(self.g_values_target1.data)
        nn.init.xavier_uniform_(self.g_values_target2.data)
        
        self.L_g = L_g
        self.deep_g = deep_g
        self.training_phase = training_phase  # "phase1" 或 "phase2"
    
    def get_current_g_values(self):
        """
        Return prompt tokens for the *current* training phase without any fusion.
        During phase1 → use g_values_target1;
        During phase2 → use g_values_target2.
        """
        if self.training_phase == "phase1":
            return self.g_values_target1
        elif self.training_phase == "phase2":
            return self.g_values_target2
        else:
            raise ValueError(f"Unknown training_phase: {self.training_phase}")

    @torch.no_grad()
    def get_fused_g_values(self, weights: torch.Tensor):
        """
        Compute input‑specific fused prompts for inference.
        Args:
            weights: Tensor of shape [batch_size, 2] giving softmax weights
                     for (target1, target2) per sample.
        Returns:
            Tensor of shape [batch_size, self.deep_g, self.L_g, self.text_model.d_model]
        """
        # Stack domain‑specific prompts: [2, deep_g, L_g, d_model]
        g_stack = torch.stack([self.g_values_target1,
                               self.g_values_target2], dim=0).to(weights.device)
        # Expand weights to match prompt dims: [batch, 2, 1, 1, 1]
        w = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # Weighted sum across domain dimension
        fused = (w * g_stack).sum(dim=1)
        return fused

    def forward(self, image: torch.Tensor, text_tokens: torch.Tensor, attn_mask: torch.Tensor, device="cuda"):
        batch_size = image.shape[0]
        # [deep_g, L_g, d_model]
        current_g_values = self.get_current_g_values()

        text_g_prompt = current_g_values.unsqueeze(0).repeat(text_tokens.size(0), 1, 1, 1).to(device)

        vision_g_prompt = self.prompt_proj(current_g_values.unsqueeze(0).repeat(batch_size, 1, 1, 1))
        
        text_out = self.text_model(text_tokens, attn_mask, text_g_prompt)
        img_out = self.vision_model(image, vision_g_prompt)
                 
        text_proj = self.text_proj(text_out)
        img_proj = self.image_proj(img_out)
        text_embed = text_proj / text_proj.norm(dim=-1, keepdim=True)
        img_embed = img_proj / img_proj.norm(dim=-1, keepdim=True)
        sim = 100 * img_embed @ text_embed.T
        return sim