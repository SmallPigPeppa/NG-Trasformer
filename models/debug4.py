import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer
from timm import create_model
from timm.models._manipulate import checkpoint_seq
import torch.nn.functional as F


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

    def forward_block_with_filter_origin(self, blk, x):
        x = blk(x)
        return x

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    #     x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    #     return x

    def forward_block_with_filter(self, blk, x):
        def _filter_attn(x):
            attn_model = blk.attn
            B, N, C = x.shape
            qkv = attn_model.qkv(x).reshape(B, N, 3, attn_model.num_heads, attn_model.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = attn_model.q_norm(q), attn_model.k_norm(k)

            if attn_model.fused_attn:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=attn_model.attn_drop.p if attn_model.training else 0.,
                )
            else:
                q = q * attn_model.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = attn_model.attn_drop(attn)
                x = attn @ v

            x = x.transpose(1, 2).reshape(B, N, C)
            x = attn_model.proj(x)
            x = attn_model.proj_drop(x)
            return x


        # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x_attn = _filter_attn(blk.norm1(x))
        x = x + self.drop_path1(self.ls1(x_attn))

        # x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x


# 初始化并加载预训练权重
def create_modified_vitb16(keep_ratio=0.8):
    model = ModifiedVisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                      qkv_bias=True, norm_layer=nn.LayerNorm, keep_ratio=keep_ratio, num_classes=1000)
    weights = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
    print(f"Loading weights {weights}")
    orig_net = create_model(weights, pretrained=True)
    state_dict = orig_net.state_dict()
    model.load_state_dict(state_dict, strict=True)
    # return orig_net
    return model, orig_net


# 创建模型
if __name__ == '__main__':
    model = create_modified_vitb16(keep_ratio=0.8)
