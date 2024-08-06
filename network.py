# Copyright © Niantic, Inc. 2022.

import logging
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

_logger = logging.getLogger(__name__)

class my_add(nn.Module):
    def __init__(self, channel, reduction=16):
        super(my_add, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # self.conv_parallel = nn.Conv2d(
        #     in_channels=channel*2, out_channels=channel*2, kernel_size=1, stride=1, padding=0)
        self.conv_single = nn.Conv2d(
            in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0)

    def selayer(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
    def ecalayer(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.conv_single(y)
        y = torch.sigmoid(y)
        return x * y.expand_as(x)

    def selayer_avgandmax(self, x):
        b, c, _, _ = x.size()
        y_1 = self.avg_pool(x).view(b, c)
        y_2 = self.max_pool(x).view(b, c)
        y = self.fc(y_1+y_2).view(b, c, 1, 1)
        return x * y

    def ecalayer_avgandmax(self, x):
        b, c, _, _ = x.size()
        y_1 = self.avg_pool(x).view(b, c, 1, 1)
        y_2 = self.max_pool(x).view(b, c, 1, 1)
        y = self.conv_single(y_1+y_2)
        y = torch.sigmoid(y)
        return x * y.expand_as(x)

    def channel_shuffle(self, x, groups=16):
        batch_size, num_channels, height, width = x.size()

        assert num_channels % groups == 0, "num_channels should be divisible by groups"
        channels_per_group = num_channels // groups
        
        # Reshape to (batch_size, groups, channels_per_group, height, width)
        x = x.view(batch_size, groups, channels_per_group, height, width)
        
        # Transpose to (batch_size, 2, groups/2, channels_per_group, height, width)
        x = x.view(batch_size, groups, 2, channels_per_group // 2, height, width)
        x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
        # x = x.transpose(1, 2).contiguous()
        
        # Reshape back to (batch_size, num_channels, height, width)
        x = x.view(batch_size, -1, height, width)

        return x

class Encoder(nn.Module):
    """
    FCN encoder, used to extract features from the input images.

    The number of output channels is configurable, the default used in the paper is 512.
    """

    def __init__(self, out_channels=512):
        super(Encoder, self).__init__()

        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)

        self.res1_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.res1_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.res1_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.res2_conv1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.res2_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.res2_conv3 = nn.Conv2d(512, self.out_channels, 3, 1, 1)

        self.res2_skip = nn.Conv2d(256, self.out_channels, 1, 1, 0)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        res = F.relu(self.conv4(x))

        x = F.relu(self.res1_conv1(res))
        x = F.relu(self.res1_conv2(x))
        x = F.relu(self.res1_conv3(x))

        res = res + x

        x = F.relu(self.res2_conv1(res))
        x = F.relu(self.res2_conv2(x))
        x = F.relu(self.res2_conv3(x))
        
        x = self.res2_skip(res) + x

        return x

class Regressor(nn.Module):
    """
    FCN architecture for scene coordinate regression.

    The network predicts a 3d scene coordinates, the output is subsampled by a factor of 8 compared to the input.
    """

    OUTPUT_SUBSAMPLE = 8

    def __init__(self, mean, num_head_blocks, use_homogeneous, num_encoder_features=512):
        """
        Constructor.

        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        num_encoder_features: Number of channels output of the encoder network.
        """
        super(Regressor, self).__init__()

        self.feature_dim = num_encoder_features
        self.encoder = Encoder(out_channels=self.feature_dim)
        self.heads = Head(mean, num_head_blocks, use_homogeneous, in_channels=self.feature_dim)




    @classmethod    
    def create_from_encoder(cls, encoder_state_dict, mean, num_head_blocks, use_homogeneous):
        """
        Create a regressor using a pretrained encoder, loading encoder-specific parameters from the state dict.


        encoder_state_dict: pretrained encoder state dictionary.
        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        """

        # Number of output channels of the last encoder layer.
        num_encoder_features = encoder_state_dict['res2_conv3.weight'].shape[0]

        # Create a regressor.

        _logger.info(f"Creating Regressor using pretrained encoder with {num_encoder_features} feature size.")
        regressor = cls(mean, num_head_blocks, use_homogeneous, num_encoder_features)

        # Load encoder weights.
        regressor.encoder.load_state_dict(encoder_state_dict)

        # Done.
        return regressor

    @classmethod
    def create_from_state_dict(cls, state_dict):
        """
        Instantiate a regressor from a pretrained state dictionary.

        state_dict: pretrained state dictionary.
        """
        # Mean is zero (will be loaded from the state dict).
        mean = torch.zeros((3,))

        # Count how many head blocks are in the dictionary.
        pattern = re.compile(r"^heads\.\d+c0\.weight$")
        num_head_blocks = sum(1 for k in state_dict.keys() if pattern.match(k))

        # Whether the network uses homogeneous coordinates.
        use_homogeneous = state_dict["heads.fc3.weight"].shape[0] == 4

        # Number of output channels of the last encoder layer.
        num_encoder_features = state_dict['encoder.res2_conv3.weight'].shape[0]

        # Create a regressor.
        _logger.info(f"Creating regressor from pretrained state_dict:"
                     f"\n\tNum head blocks: {num_head_blocks}"
                     f"\n\tHomogeneous coordinates: {use_homogeneous}"
                     f"\n\tEncoder feature size: {num_encoder_features}")
        regressor = cls(mean, num_head_blocks, use_homogeneous, num_encoder_features)

        # Load all weights.
        regressor.load_state_dict(state_dict)

        # Done.
        return regressor

    @classmethod
    def create_from_split_state_dict(cls, encoder_state_dict, head_state_dict):
        """
        Instantiate a regressor from a pretrained encoder (scene-agnostic) and a scene-specific head.

        encoder_state_dict: encoder state dictionary
        head_state_dict: scene-specific head state dictionary
        """
        # We simply merge the dictionaries and call the other constructor.
        merged_state_dict = {}

        for k, v in encoder_state_dict.items():
            merged_state_dict[f"encoder.{k}"] = v

        for k, v in head_state_dict.items():
            merged_state_dict[f"heads.{k}"] = v

        return cls.create_from_state_dict(merged_state_dict)

    def load_encoder(self, encoder_dict_file):
        """
        Load weights into the encoder network.
        """
        self.encoder.load_state_dict(torch.load(encoder_dict_file))

    def get_features(self, inputs):
        return self.encoder(inputs)
    
    def get_scene_coordinates(self, features):
        return self.heads(features)
    
    def get_head_param(self):
        return sum(p.numel() for p in self.heads.parameters())

    def forward(self, inputs):
        """
        Forward pass.
        """
        features = self.get_features(inputs)
        return self.get_scene_coordinates(features)
        # scene_coordinates_B3HW = self.get_scene_coordinates(features)
        # heads_params = sum(p.numel() for p in self.heads.parameters())
        # return heads_params, scene_coordinates_B3HW


class Head(nn.Module):
    def __init__(self,
                 mean,
                 num_head_blocks,
                 use_homogeneous,
                 homogeneous_min_scale=0.01,
                 homogeneous_max_scale=4.0,
                 in_channels=512):
        super(Head, self).__init__()

        self.use_homogeneous = use_homogeneous 
        self.in_channels = in_channels  # Number of encoder features.
        self.head_channels = 512  # Hardcoded.
        self.dense_channels = [256,128]

        # self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU()
        self.relu = nn.PReLU()

        self.my_add = my_add(channel=in_channels)

        self.head_skip = nn.Identity() if self.in_channels == self.head_channels else nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.head_channels, kernel_size=1, stride=1, padding=0)

        self.res3_conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.head_channels, kernel_size=1, stride=1, padding=0)
        self.res3_conv2 = nn.Conv2d(in_channels=self.head_channels, out_channels=self.head_channels, kernel_size=1, stride=1, padding=0)
        self.res3_conv3 = nn.Conv2d(in_channels=self.head_channels, out_channels=self.head_channels, kernel_size=1, stride=1, padding=0)

        self.res_blocks = []

        for block in range(num_head_blocks):
            self.res_blocks.append((
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.dense_channels[0], kernel_size=1, stride=1, padding=0),
                nn.Conv2d(in_channels=self.in_channels+self.dense_channels[0], out_channels=self.dense_channels[1], kernel_size=1, stride=1, padding=0),
                nn.Conv2d(in_channels=self.in_channels+self.dense_channels[0]+self.dense_channels[1], out_channels=self.head_channels, kernel_size=1, stride=1, padding=0),
            ))

            super(Head, self).add_module(str(block) + 'c0', self.res_blocks[block][0])
            super(Head, self).add_module(str(block) + 'c1', self.res_blocks[block][1])
            super(Head, self).add_module(str(block) + 'c2', self.res_blocks[block][2])

        self.fc1 = nn.Conv2d(in_channels=self.head_channels, out_channels=self.head_channels, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv2d(in_channels=self.head_channels, out_channels=self.head_channels, kernel_size=1, stride=1, padding=0)
        

        if self.use_homogeneous:  
            self.fc3 = nn.Conv2d(self.head_channels, 4, 1, 1, 0)

            # Use buffers because they need to be saved in the state dict.
            self.register_buffer("max_scale", torch.tensor([homogeneous_max_scale]))
            self.register_buffer("min_scale", torch.tensor([homogeneous_min_scale]))
            self.register_buffer("max_inv_scale", 1. / self.max_scale)
            self.register_buffer("h_beta", math.log(2) / (1. - self.max_inv_scale))
            self.register_buffer("min_inv_scale", 1. / self.min_scale)
        else:
            self.fc3 = nn.Conv2d(self.head_channels, 3, 1, 1, 0)

        self.register_buffer("mean", mean.clone().detach().view(1, 3, 1, 1))


    def forward(self, input):

        # input = self.my_add.ecalayer(input)
        input_sc = self.head_skip(input)

        # res = self.my_add.ecalayer_avgandmax(res)
        x_res_1 = self.relu(self.res3_conv1(input_sc))
        x_res_2 = self.relu(self.res3_conv2(x_res_1))
        x_res_3 = self.relu(self.res3_conv3(x_res_2))
        
        x_res = input_sc + x_res_3
        # res_1 = self.my_add.ecalayer_avgandmax(res_1)

        for res_block in self.res_blocks:
            x_dense_1 = self.relu(res_block[0](input_sc))
            x_dense_2 = self.relu(res_block[1](torch.cat((input_sc,x_dense_1),dim=1)))
            x_dense = self.relu(res_block[2](torch.cat((input_sc,x_dense_1,x_dense_2),dim=1)))

            # dense_1 = input_sc + x_dense
            # res_2 = self.my_add.ecalayer_avgandmax(res_2)

        # res_dense = self.my_add.ecalayer_avgandmax(x_res_3 + x_dense) + input_sc
        # res_dense = x_res_3 + x_dense + input_sc
        res_dense = x_res + x_dense  
        res_dense = self.my_add.ecalayer_avgandmax(res_dense)
        sc = self.relu(self.fc1(res_dense))
        sc = self.relu(self.fc2(sc))
        sc = self.fc3(sc)

        if self.use_homogeneous:
            h_slice = F.softplus(sc[:, 3, :, :].unsqueeze(1), beta=self.h_beta.item()) + self.max_inv_scale
            h_slice.clamp_(max=self.min_inv_scale)
            sc = sc[:, :3] / h_slice
        sc += self.mean

        return sc