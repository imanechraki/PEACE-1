from torch import nn
import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download

# login('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',add_to_git_credential=True)  # login with your User Access Token, found at https://huggingface.co/settings/tokens

local_dir = "/gpfs/workdir/chrakii/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/" 
# hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)



class UniFeatures(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )

        self.model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
        self.t = transforms.Compose(
            [transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        )

        self.model.eval()

    def __call__(self, x):

        # get the features
        with torch.inference_mode():
            features = self.model(self.t(x))
      
        return features


