
from torch import nn
import torch
import monai

class SequenceUnetBackbone(nn.Module):
    def __init__(self, spatial_dims, out_channels, channels, strides, norm, num_res_units, bias, dropout):
        super().__init__()
        self.backbone = monai.networks.nets.UNet(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            norm=norm,
            num_res_units=num_res_units,
            bias=bias,
            dropout=dropout)
        
        self.conv = nn.Conv3d(out_channels*2, out_channels, kernel_size=3, padding="same")
        self.final_conv = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding="same")
        

    def forward(self, x):
        features = None
        for s in range(x.shape[1]):
            # B, W, H, D -> B, 1, W, H, D
            input_timestamp = torch.unsqueeze(x[:, s], dim=1)
            # B, C, W, H, D
            current_features = self.backbone(input_timestamp)
            if features is None:
                features = current_features
            else:
                # Concatenate the features
                # B, 2*C, W, H, D
                features = torch.cat((features, current_features), dim=1)
                # B, 2*C, W, H, D -> B, C, W, H, D
                features = self.conv(features)
        features = self.final_conv(features)
        return features
class SequenceUNet(nn.Module):
    def __init__(self, crop_size, spatial_dims, out_channels, channels, strides, norm, num_res_units, bias, dropout, tap_mode=False):
        super().__init__()
        self.backbone = SequenceUnetBackbone(
            spatial_dims=spatial_dims,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            norm=norm,
            num_res_units=num_res_units,
            bias=bias,
            dropout=dropout)
        
        self.tap_mode = tap_mode
        if self.tap_mode:
            self.pool = torch.nn.AdaptiveAvgPool3d((4,4,4))
            self.flatten_layer = torch.nn.Flatten()
            # num_features_in = 2 * 2 * 2 * out_channels
            # self.batch_norm = nn.BatchNorm1d(num_features_in)
            # self.pool = torch.nn.AdaptiveAvgPool3d((1))
            self.classify_head = nn.Linear(out_channels*4*4*4, 1)
        else:
            self.classify_head = self.build_head(in_channels=out_channels)

    def build_head(self, in_channels):
        mod = nn.Sequential()
        mod.append(nn.Conv3d(in_channels, 1, kernel_size=3, padding="same"))
        return mod
       
    def forward(self, x):
        # x is a sequence of shape (B, S, W, H, D)
        # compute features for the whole sequence
        features = self.backbone(x)
        
        if self.tap_mode:
            z = self.pool(features)
            z = self.flatten_layer(z)
            # z = self.batch_norm(z)
            logits = self.classify_head(z)
            return logits, features
        else:
            # Perform conv on the summed features to perform classification
            heatmap = self.classify_head(features)
            
            return {"heatmaps": [heatmap]}