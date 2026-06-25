
from torch import nn
import torch
import monai

class UNet(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, channels, strides, norm, num_res_units, bias, dropout, return_tap=False):
        super().__init__()
        self.backbone = monai.networks.nets.UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            norm=norm,
            num_res_units=num_res_units,
            bias=bias,
            dropout=dropout)
        self.return_tap = return_tap

        # if self.return_tap:
            # self.bottom_layer = self.get_deepest_submodule(self.backbone.model[1])
            # self.bottom_layer.register_forward_hook(self.hook_features)
            # self.flatten_layer = nn.Flatten()
            # # num_features_in = 2*2*2*512
            # num_features_in = 512
            # self.batch_norm = nn.BatchNorm1d(num_features_in)
            # self.classif_head = nn.Linear(num_features_in, 1)
        
    # def hook_features(self, model, input, output):
    #     self.features_output = output

    # def get_deepest_submodule(self, node):
    #     if hasattr(node, "submodule"):
    #         if isinstance(node.submodule, torch.nn.Sequential):
    #             return self.get_deepest_submodule(node.submodule[1])
    #         else:
    #             return node.submodule
    #     return node

    def unet_features(self, x):
        return self.backbone(x)
        # return self.features_output # Stored by the hook

    def forward(self, x):
        if self.return_tap: # Return the classification on the lowest point of the unet
            features = self.unet_features(x)
            # Flatten
            # x = torch.reshape(features, [features.shape[0], -1])
            # x = self.flatten_layer(x)
            # x = self.batch_norm(x)
            # x = self.classif_head(x)
            # x = nn.functional.sigmoid(x)
            return features
        else: # Return the whole unet
            x = self.backbone(x)
            return x