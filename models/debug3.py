import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer


class ModifiedVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super(ModifiedVisionTransformer, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = self.forward_block_with_filter(blk, x)

        return self.norm(x[:, 0])

    def forward_block_with_filter(self, blk, x):
        x = blk.norm1(x)
        qkv = blk.attn.qkv(x).reshape(x.shape[0], x.shape[1], 3, blk.attn.num_heads, blk.attn.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
        attn = attn.softmax(dim=-1)

        # 筛选前80%的注意力权重
        sorted_attn, indices = torch.sort(attn, dim=-1, descending=True)
        threshold_index = int(sorted_attn.shape[-1] * 0.8)
        mask = torch.arange(sorted_attn.shape[-1]).expand_as(sorted_attn) < threshold_index
        mask = mask.to(attn.device)

        # 将低于阈值的权重设置为0
        attn[~mask] = 0
        attn = attn / attn.sum(dim=-1, keepdim=True)  # 重新归一化权重

        x = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], blk.attn.num_heads * blk.attn.head_dim)
        x = blk.attn.proj(x)

        x = blk.drop_path1(x) + x
        x = blk.norm2(x)
        x = blk.mlp(x)
        x = blk.drop_path2(x) + x

        return x


# 使用修改后的模型
model = ModifiedVisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                  qkv_bias=True, norm_layer=nn.LayerNorm)

