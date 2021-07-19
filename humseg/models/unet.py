import torch
import torch.nn as nn
from torchvision.models import resnet34

from .fpa import FeaturePyramidAttention


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):

        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")

        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        return x


class Unet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        classes: int = 1,
    ):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.classes = classes

        resnet34_model = resnet34(pretrained=True, progress=False)

        self.encoder_layers = nn.ModuleList(
            [
                nn.Sequential(
                    resnet34_model.conv1,
                    resnet34_model.bn1,
                    resnet34_model.relu,
                ),
                nn.Sequential(resnet34_model.maxpool, resnet34_model.layer1),
                resnet34_model.layer2,
                resnet34_model.layer3,
                resnet34_model.layer4,
            ]
        )

        encoder_output_channels = [64, 64, 128, 256, 512]

        skip_channels = encoder_output_channels[:-1][::-1] + [0]

        decoder_input_channels = [512, 256, 128, 64, 32]

        decoder_output_channels = [256, 128, 64, 32, 16]

        self.decoder_layers = nn.ModuleList(
            [
                DecoderBlock(in_channel, skip_channel, out_channel)
                for in_channel, skip_channel, out_channel in list(
                    zip(
                        decoder_input_channels,
                        skip_channels,
                        decoder_output_channels,
                    )
                )
            ]
        )

        self.head = nn.Conv2d(
            decoder_output_channels[-1],
            self.classes,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        encoder_features = []

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encoder_features.append(x)

        skip_connections = encoder_features[:-1]
        skip_connections = skip_connections[::-1]

        for i, decoder_block in enumerate(self.decoder_layers):
            skip = skip_connections[i] if i < len(skip_connections) else None
            x = decoder_block(x, skip)

        return self.head(x)


class UnetFPA(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        classes: int = 1,
    ):
        super(UnetFPA, self).__init__()
        self.in_channels = in_channels
        self.classes = classes

        resnet34_model = resnet34(pretrained=True, progress=False)

        self.encoder_layers = nn.ModuleList(
            [
                nn.Sequential(
                    resnet34_model.conv1,
                    resnet34_model.bn1,
                    resnet34_model.relu,
                ),
                nn.Sequential(resnet34_model.maxpool, resnet34_model.layer1),
                resnet34_model.layer2,
                resnet34_model.layer3,
            ]
        )

        encoder_output_channels = [64, 64, 128, 256]

        skip_channels = encoder_output_channels[:-1][::-1] + [0]

        decoder_input_channels = [256, 128, 64, 32]

        decoder_output_channels = [128, 64, 32, 16]

        self.decoder_layers = nn.ModuleList(
            [
                DecoderBlock(in_channel, skip_channel, out_channel)
                for in_channel, skip_channel, out_channel in list(
                    zip(
                        decoder_input_channels,
                        skip_channels,
                        decoder_output_channels,
                    )
                )
            ]
        )

        self.fpa = FeaturePyramidAttention(256)

        self.head = nn.Conv2d(
            decoder_output_channels[-1],
            self.classes,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        encoder_features = []

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encoder_features.append(x)

        skip_connections = encoder_features[:-1]
        skip_connections = skip_connections[::-1]

        x = self.fpa(x)

        for i, decoder_block in enumerate(self.decoder_layers):
            skip = skip_connections[i] if i < len(skip_connections) else None
            x = decoder_block(x, skip)

        return self.head(x)
