# vit_base_patch16_224.augreg_in21k_ft_in1k

# def vit_base_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
#     """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
#     """
#     model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
#     model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model

from vit import vit_base_patch16_224

m = vit_base_patch16_224(pretrained=True)
print(m)
