import monai
from torch import nn

class SwinUNETRWrapper(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, depths, num_heads, feature_size, 
                 norm_name, drop_rate, attn_drop_rate, dropout_path_rate, normalize, spatial_dims, 
                 downsample, use_v2, use_checkpoint):
        super().__init__()
        self.backbone= monai.networks.nets.SwinUNETR(
            img_size = img_size,
            in_channels = in_channels,
            out_channels = out_channels,
            depths = depths,
            num_heads = num_heads,
            feature_size = feature_size,
            norm_name = norm_name,
            drop_rate = drop_rate,
            attn_drop_rate = attn_drop_rate,
            dropout_path_rate = dropout_path_rate,
            normalize = normalize,
            spatial_dims = spatial_dims,
            downsample = downsample,
            use_v2 = use_v2,
            use_checkpoint = use_checkpoint
        )
        
    def forward(self, x):
        return {"heatmaps": [self.backbone(x)]}