import os
from typing import Tuple, Optional, Union
from safetensors.torch import load_file
from timm.models import ByoBlockCfg, ByoModelCfg, ByobNet
from timm.models._registry import register_model
from timm.models._builder import build_model_with_cfg

__all__ = ['ResNet32', 'resnet32CI']

resnet32_cfg = ByoModelCfg(
    blocks=(
        ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
        ByoBlockCfg(type='bottle', d=3, c=512, s=2, gs=0, br=0.25),
        ByoBlockCfg(type='bottle', d=3, c=1536, s=2, gs=0, br=0.25),
        ByoBlockCfg(type='bottle', d=2, c=1536, s=2, gs=0, br=0.25),
    ),
    stem_chs=64,
    stem_type='tiered',
    stem_pool='',
    num_features=0,
    act_layer='silu',
)


class ResNet32(ByobNet):

    @classmethod
    def create(self):
        return ResNet32(cfg=resnet32_cfg)

    def forward(self, x, label_set=None):
        x = super().forward(x)
        if label_set is not None:
            x = x[:, label_set]
        return x
    
    def freeze(self):
        for parameter in self.parameters():
            parameter.requires_grad = False


@register_model
def resnet32CI(pretrained=False, **kwargs) -> ResNet32:
    checkpoint_dirs = [
        os.path.join(os.path.expanduser('~'), '.cache/huggingface/models--timm--resnet32ts.ra2_in1k/snapshots'),
        os.path.join(os.path.expanduser('~'), '.cache/huggingface/hub/models--timm--resnet32ts.ra2_in1k/snapshots')
    ]
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            for folder in os.listdir(checkpoint_dir):
                checkpoint_path = os.path.join(checkpoint_dir, folder, 'model.safetensors')
                if os.path.exists(checkpoint_path):
                    model = ResNet32.create()
                    model.load_state_dict(load_file(checkpoint_path))
                    return model
    return build_model_with_cfg(
        ResNet32, 'resnet32ts', pretrained,
        model_cfg=resnet32_cfg,
        feature_cfg=dict(flatten_sequential=True),
        **kwargs)


if __name__ == '__main__':
    model = resnet32CI(pretrained=True)
    import pdb; pdb.set_trace()
