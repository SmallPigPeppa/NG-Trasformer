import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer
from timm import create_model
from timm.models._manipulate import checkpoint_seq


class ModifiedVisionTransformer(VisionTransformer):
    def __init__(self, *args, keep_ratio=0.8, **kwargs):
        super(ModifiedVisionTransformer, self).__init__(*args, **kwargs)
        self.keep_ratio = keep_ratio


    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            for block in self.blocks:
                x = self.forward_block_with_filter(block, x)
        x = self.norm(x)
        return x

    def forward_block_with_filter(self, blk, x):
        # Apply normalization
        x_norm = blk.norm1(x)

        # Compute attention with some probabilities set to zero
        B, N, C = x_norm.shape
        qkv = blk.attn.qkv(x_norm).reshape(B, N, 3, blk.attn.num_heads, blk.attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = blk.attn.q_norm(q), blk.attn.k_norm(k)

        q = q * blk.attn.scale
        attn = q @ k.transpose(-2, -1)

        # Softmax to get attention probabilities
        attn = attn.softmax(dim=-1)

        # Set the smallest 20% of the attention probabilities to 0
        k_threshold = int(attn.numel() * self.keep_ratio)
        threshold_values, _ = torch.topk(attn.view(-1), k_threshold, largest=False)
        threshold_value = threshold_values.max()
        attn = torch.where(attn < threshold_value, torch.tensor(0.0).to(attn.device), attn)

        attn = blk.attn.attn_drop(attn)
        x_attn = attn @ v

        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
        x_attn = blk.attn.proj(x_attn)
        x_attn = blk.attn.proj_drop(x_attn)

        # Add residual and apply drop path
        x = x + blk.drop_path1(blk.ls1(x_attn))

        # MLP part
        x_norm2 = blk.norm2(x)
        x_mlp = blk.mlp(x_norm2)
        x = x + blk.drop_path2(blk.ls2(x_mlp))

        return x


# 初始化并加载预训练权重
def create_modified_vitb16(keep_ratio=0.8):
    model = ModifiedVisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                      qkv_bias=True, norm_layer=nn.LayerNorm, keep_ratio=keep_ratio,num_classes=1000)
    weights='vit_base_patch16_224.augreg2_in21k_ft_in1k'
    print(f"Loading weights {weights}")
    orig_net = create_model(weights, pretrained=True)
    state_dict = orig_net.state_dict()
    model.load_state_dict(state_dict, strict=True)
    # return orig_net
    return model


# 创建模型
if __name__ == '__main__':
    model = create_modified_vitb16(keep_ratio=0.8)
