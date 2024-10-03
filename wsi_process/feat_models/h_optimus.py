from torch import nn

import functools

import timm
import torch
from torchvision import transforms 




class HOptimusFeatures(nn.Module):
    def __init__(self):
        super().__init__()

        #resnet = resnet50(weights="IMAGENET1K_V2")
        PATH_TO_CHECKPOINT = "/gpfs/workdir/chrakii/checkpoint.pth"  # Path to the downloaded checkpoint.

        params = {
            'patch_size': 14, 
            'embed_dim': 1536, 
            'depth': 40, 
            'num_heads': 24, 
            'init_values': 1e-05, 
            'mlp_ratio': 5.33334, 
            'mlp_layer': functools.partial(
                timm.layers.mlp.GluMlp, act_layer=torch.nn.modules.activation.SiLU, gate_last=False
            ), 
            'act_layer': torch.nn.modules.activation.SiLU, 
            'reg_tokens': 4, 
            'no_embed_class': True, 
            'img_size': 224, 
            'num_classes': 0, 
            'in_chans': 3
        }

        self.model = timm.models.VisionTransformer(**params)
        self.model.load_state_dict(torch.load(PATH_TO_CHECKPOINT, map_location="cpu"))
        self.model.eval()

        self.t = transforms.Compose(
            [transforms.Normalize(
        mean=(0.707223, 0.578729, 0.703617), 
        std=(0.211883, 0.230117, 0.177517)
    ),]
        )

    def __call__(self, x):
        # We recommend using mixed precision for faster inference.
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.inference_mode():
                x = self.model(self.t(x))
        
        return x


