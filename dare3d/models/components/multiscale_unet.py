
from torch import nn
import torch
import monai

class MultiScaleUNet(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, channels, strides, norm, num_res_units, bias, dropout, output_names, downsample_factors=[1]):
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

        # Example: ["model.0.conv"]
        if len(downsample_factors) == 1:
            print("Using only 1 output scale")
            self.output_names = []
        else:
            self.output_names = output_names

        print(f"Using output names: {self.output_names}")
        # Now {"model.0.conv": "heatmap0"}
        self.output_names_dict = {
            name: 
                {
                    "index": f"heatmap{i+1}",
                    "channels": channels[i],
                }
             for i, name in enumerate(self.output_names)}
        self.conv_modules = nn.ModuleList()
        self.build_output_models(n_conv=1)
                
        self.output_layers = {}
        num_registered = 0
        for name, mod in self.backbone.named_modules():
            if name not in self.output_names:
                continue
            num_registered += 1
            mod.register_forward_hook(self.hook_level(self.output_names_dict[name]))
        assert num_registered == len(self.output_names), print(self.backbone)

    def build_output_models(self, n_conv, bn=True):
        for name, values in self.output_names_dict.items():
            if values["index"] == "heatmap0":
                continue
            in_channels = values["channels"]
            mod = nn.Sequential()
            for i in range(n_conv):
                out_channels = in_channels // 2
                if i == n_conv - 1:
                    mod.append(nn.Conv3d(in_channels, 1, kernel_size=3, padding="same"))
                else:
                    mod.append(
                        nn.Sequential(
                            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding="same"),
                            nn.BatchNorm3d(out_channels)
                        )
                    )
                in_channels = out_channels
            self.conv_modules.append(mod)
            self.output_names_dict[name]["module"] = len(self.conv_modules) - 1
    
    def hook_level(self, name):
        def hook(model, input, output):
            # Forward pass through classification layer
            if name["index"] != "heatmap0":
                mod = self.conv_modules[name["module"]]
                output = mod(output)
            self.output_layers[name["index"]] = output
        return hook

    def get_deepest_submodule(self, node):
        if hasattr(node, "submodule"):
            if isinstance(node.submodule, torch.nn.Sequential):
                return self.get_deepest_submodule(node.submodule[1])
            else:
                return node.submodule
        return node

    def unet_features(self, x):
        _ = self.backbone(x)
        return self.features_output # Stored by the hook

    def forward(self, x):
        heatmaps = [self.backbone(x)]
        if len(self.output_names) > 0:
            heatmaps += [self.output_layers[f"heatmap{i+1}"] for i in range(len(self.output_names))]
        return {"heatmaps": heatmaps}