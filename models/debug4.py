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
        x = blk(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

    def forward_block_with_filter_old(self, blk, x):
        # Apply normalization
        x_norm1 = blk.norm1(x)
        x_attn = blk.attn(x_norm1)
        x = x + self.drop_path1(self.ls1(x_attn))

        # MLP part
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
