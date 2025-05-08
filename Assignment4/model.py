# model.py: Implements the VQA model (Parts 4–7)

import torch
import torch.nn as nn
import torchvision.models as models


class VQAModel(nn.Module):
    def __init__(self, vocab_size, num_classes, max_len=30, embed_dim=768):
        super(VQAModel, self).__init__()

        # === Part 4: Image Encoder ===
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]  # Remove avgpool and fc
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.linear_proj = nn.Linear(2048, embed_dim)

        # === Part 5: Text Encoder ===
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, max_len + 1, embed_dim))  # +1 for CLS
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # === Part 6: Cross Attention ===
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)

        # === Part 7: Classifier ===
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 500),
            nn.ReLU(),
            nn.Linear(500, num_classes)
        )

    
    def forward(self, images, questions, attn_masks):
        B = images.size(0)
        device = images.device
    
        # === Image Features ===
        feat_map = self.resnet(images)  # [B, 2048, H, W]
        H, W = feat_map.size(2), feat_map.size(3)
        feat_map = feat_map.view(B, 2048, -1).permute(0, 2, 1)  # [B, H*W, 2048]
        img_feats = self.linear_proj(feat_map)  # [B, H*W, 768]
    
        # === Text Features ===
        x = self.token_embed(questions)  # [B, max_len, 768]
        cls = self.cls_token.repeat(B, 1, 1)  # [B, 1, 768]
        x = torch.cat([cls, x], dim=1)  # [B, max_len+1, 768]
    
        # Use attn_masks from input (shape [B, max_len])
        pad_mask = (attn_masks == 0)
        pad_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=device), pad_mask], dim=1)  # [B, L+1]
    
        # Permute x to [seq_len, batch, embed_dim] for transformer
        x = x.permute(1, 0, 2)  # [max_len+1, B, 768]
        x = x + self.pos_embed.permute(1, 0, 2)[:, :x.size(0), :]  # Adjust positional embedding
        x = self.text_encoder(x, src_key_padding_mask=pad_mask)  # [max_len+1, B, 768]
        x = x.permute(1, 0, 2)  # Back to [B, max_len+1, 768]
    
        cls_token_encoded = x[:, 0:1, :]  # [B, 1, 768]
    
        # === Cross Attention ===
        attn_out, _ = self.cross_attention(
            query=cls_token_encoded,
            key=img_feats,
            value=img_feats
        )
    
        # === Classifier ===
        logits = self.mlp(attn_out.squeeze(1))
        return logits
        
