
from torch import nn
import torch
import monai

from dare3d.models.components.multiscale_unet import MultiScaleUNet

class SegResMultiScaleUNet(MultiScaleUNet):
    def __init__(self, hook_layer_name, regression_in_channels, n_upsample, **kwargs):
        super().__init__(**kwargs)

        # The name of the layer from which we will regress the parameters
        self.hook_layer_name = hook_layer_name
        self.regression_in_channels = regression_in_channels
        self.n_upsample = n_upsample
        
        # model.1.submodule.1.submodule.2.1
        
        # Build the modules to regress the parameters
        self.build_regression_modules()

        for name, mod in self.backbone.named_modules():
            if name != self.hook_layer_name:
                continue
            mod.register_forward_hook(self.hook_regression())

    def build_block(self, n_upsample, in_channels, out_channels):
        final_out_channels = out_channels
        mod = nn.Sequential()
        for _ in range(n_upsample):
            out_channels = in_channels // 2
            mod.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1), bias=False),
                    nn.BatchNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.PReLU(num_parameters=1)
                )
            )
            in_channels = out_channels
        mod.append(
            nn.Conv3d(in_channels, final_out_channels, kernel_size=3, padding="same")
        )
        return mod

    def build_regression_modules(self):
        in_channels = self.regression_in_channels
        
        # Build two modules
        # One for the length (2,W,H,D)
        self.length_mod = self.build_block(self.n_upsample, in_channels, 2)
        # One for the quaternion        
        self.rotmat_mod = self.build_block(self.n_upsample, in_channels, 18)
    
    def hook_regression(self):
        def hook(model, input, output):
            # Forward pass through classification layer
            length_output = self.length_mod(output)
            rotmat_output = self.rotmat_mod(output)
            self.output_layers["length"] = length_output
            self.output_layers["rotmat"] = rotmat_output
        return hook

    def forward(self, x):
        data_dict = super().forward(x)
        data_dict["length"] = self.output_layers["length"]
        data_dict["rotmat"] = self.output_layers["rotmat"]
        return data_dict