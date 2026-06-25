import torch
from torch import nn
import monai


class EfficientNetNoHead(monai.networks.nets.EfficientNetBNFeatures):
    def forward(self, inputs: torch.Tensor):
        # Stem
        x = self._conv_stem(self._conv_stem_padding(inputs))
        x = self._swish(self._bn0(x))
        # Blocks
        x = self._blocks(x)
        # Head
        x = self._conv_head(self._conv_head_padding(x))
        x = self._swish(self._bn1(x))

        x = x.flatten(start_dim=1)
        return x

class Regression3dCNN(nn.Module):
    def __init__(
        self,
        im_size,
        input_channels,
        len_activation=nn.Sigmoid(),
        angle_vec_size=9,
        angle_activation=nn.Identity(),
        backbone_out_dim=2560
    ) -> None:
        super().__init__()
        self.n_channels = len(input_channels)
        self.angle_activation = angle_activation
        self.len_activation = len_activation

        self.backbone = EfficientNetNoHead(
            "efficientnet-b7",
            spatial_dims=3,
            in_channels= self.n_channels,
        )

        self.head1_len, self.head1_angle = self.create_output_head(backbone_out_dim, self.len_activation, self.angle_activation, angle_vec_size)
        self.head2_len, self.head2_angle = self.create_output_head(backbone_out_dim, self.len_activation, self.angle_activation, angle_vec_size)

    def create_output_head(self, input_vec, len_act, angle_act, angle_vec_size):
        # Create length regression
        len_head = nn.Sequential(nn.Linear(input_vec, 1), len_act)

        # Create angle regression
        angle_head = nn.Sequential(nn.Linear(input_vec, angle_vec_size), angle_act)
        return len_head, angle_head

    def forward(self, x):
        features = self.backbone(x)
               
        len1 = self.head1_len(features)
        len2 = self.head2_len(features)
        
        angle1 = self.head1_angle(features)
        angle2 = self.head2_angle(features)
        
        return {
            "head1": {"len": len1, "angle": angle1},
            "head2": {"len": len2, "angle": angle2}
            }