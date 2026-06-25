import torch
from torch import nn

class ResidualBlock(nn.Module):
    
    def __init__(self, in_filters, out_filters) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_filters, out_filters, kernel_size=(3, 3, 3), padding="same")
        self.conv2 = nn.Conv3d(out_filters, out_filters, kernel_size=(3, 3, 3), padding="same")
        self.conv3 = nn.Conv3d(out_filters, out_filters, kernel_size=(3, 3, 3), padding="same")
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool3d(kernel_size=(2,2,2))
        self.norm = nn.BatchNorm3d(out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        fx = self.conv2(x)
        fx = self.relu(fx)
        fx = self.conv3(fx)

        x = torch.add(x, fx)
        x = self.relu(x)
        x = self.norm(x)
        return x        

class RegressionNet(nn.Module):
    
    def __init__(self,input_channels,
        im_size,
        len_activation=nn.Sigmoid(),
        angle_vec_size=9,
        angle_activation=nn.Identity(),
        n_stages=3,
        start_filters=8):
        super().__init__()
        
        self.n_channels = len(input_channels)
        self.angle_activation = angle_activation
        self.len_activation = len_activation

        self.blocks = nn.ModuleList()
        in_filters = self.n_channels
        for i in range(n_stages):
            if i == 0:
                out_filters = start_filters
            self.blocks.append(ResidualBlock(in_filters, out_filters))
            in_filters = out_filters
            out_filters *= 2

        last_im_size = int(im_size / (2 ** n_stages))
        im_size_flatten = last_im_size ** 3
        outdim = int(in_filters * im_size_flatten)
        
        self.head1_len, self.head1_angle = self.create_output_head(outdim, self.len_activation, self.angle_activation, angle_vec_size)
        self.head2_len, self.head2_angle = self.create_output_head(outdim, self.len_activation, self.angle_activation, angle_vec_size)

    def create_output_head(self, input_vec, len_act, angle_act, angle_vec_size):
        # Create length regression
        len_head = nn.Sequential(nn.Linear(input_vec, 1), len_act)

        # Create angle regression
        angle_head = nn.Sequential(nn.Linear(input_vec, angle_vec_size), angle_act)
        return len_head, angle_head

    def forward(self, x):
        features = x
        for block in self.blocks:
            features = block(features)

        features = features.flatten(start_dim=1)

        len1 = self.head1_len(features)
        len2 = self.head2_len(features)
        
        angle1 = self.head1_angle(features)
        angle2 = self.head2_angle(features)
        
        return {
            "head1": {"len": len1, "angle": angle1},
            "head2": {"len": len2, "angle": angle2}
            }