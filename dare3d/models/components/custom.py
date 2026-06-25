
from torch import nn
import torch
import monai

class CustomNet(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, channels, strides, norm, num_res_units, bias):
        super().__init__()
        self.backbone = monai.networks.nets.UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            norm=norm,
            num_res_units=num_res_units,
            bias=bias)

        # Take
        
        
        self.bottom_layer = self.get_deepest_submodule(self.backbone)
        self.classif_head = nn.Linear(, 1)

    def get_deepest_submodule(self, node):
        if hasattr(node, "submodule"):
            if len(node.submodule) >= 2:
                return self.get_deepest_submodule(node.submodule[1])
        return node

    def unet_features(self, x):
        x = self.backbone.model[0](x)

    def forward(self, x):
        features = self.unet_features(x)
        x = self.classif_head(x)
        return x