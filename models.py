import torch
import torch.nn as nn
import torchvision

from enum import Enum
from transformers import CLIPVisionModel

class Pretrain(Enum):
    NO_PRETRAIN = 0
    PATH = 1
    URL = 2


def hook_fn(module, input, output, name, d):
    d.update({name: output})


class GeoFinderV1(nn.Module):

    def __init__(
        self,
        num_classes=42,
        in_channels=6,
        pretrained=Pretrain.NO_PRETRAIN,
        pretrain_url="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        **kwargs
    ) -> None:

        super(GeoFinderV1, self).__init__(**kwargs)
        self.adapter = nn.Conv2d(
            in_channels=in_channels,
            out_channels=3,
            kernel_size=1,
            stride=1
        )
        self.backbone = torchvision.models.resnet50()
        if pretrained == Pretrain.URL:
            state_dict = torch.hub.load_state_dict_from_url(pretrain_url)
            self.backbone.load_state_dict(state_dict)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(
            in_features=in_features,
            out_features=num_classes,
            bias=False
        )
        self.layers = dict(self.backbone.named_modules())
        self.intermediates = {}

    def enable_layer(self, layer_name):

        self.layers[layer_name].register_forward_hook(lambda module, input, output, name=layer_name, d=self.intermediates: hook_fn(module, input, output, name, d))

    def forward(self, x):
        return self.backbone(self.adapter(x))


class GeoFinderV2(nn.Module):

    def __init__(
        self,
        num_classes=42,
        in_channels=3,
        features_base = 32,
        **kwargs
    ) -> None:
        super(GeoFinderV2, self).__init__(**kwargs)

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, features_base, kernel_size=4, stride=2, padding=1), # 32x64x64
            nn.LeakyReLU(0.2),
            self._block(features_base, features_base * 2, 4, 2, 1), # 64x32x32
            self._block(features_base * 2, features_base * 4, 4, 2, 1), # 128x16x16
            self._block(features_base * 4, features_base * 8, 4, 2, 1), # 256x8x8
            self._block(features_base * 8, features_base * 16, 4, 2, 1), # 512x4x4
            self._block(features_base * 16, features_base * 32, 4, 2, 1), # 1024x4x4
        )

        self.head = nn.Sequential(
            nn.Linear(features_base * 32, features_base * 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(features_base * 2, num_classes)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):

        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.1)
        )

    def _get_embedding(self, x):
        features = self.backbone(x)
        feature_embeds = torch.mean(features, dim=(-2,-1))
        return feature_embeds

    def forward(self, x):
        if len(x.shape) == 4:
            embeds = self._get_embedding(x)
        else:
            embeds = torch.stack(
                [self._get_embedding(x[:,:,idx,:,:]) for idx in range(x.shape[2])],
                dim=-1
            ).mean(dim=-1)

        return self.head(embeds)

class GeoFinderV3(nn.Module):

    def __init__(
        self,
        num_classes=42,
        in_channels=3,
        features_base = 32,
        **kwargs
    ) -> None:
        super(GeoFinderV3, self).__init__(**kwargs)

        self.backbone = nn.Sequential(
            nn.Conv3d(in_channels, features_base, kernel_size=3, stride=1, padding=1), # 32x2x128x128
            nn.LeakyReLU(0.2),
            self._block(features_base, features_base * 2, (3,4,4), (1,2,2), 1), # 64x2x64x64
            self._block(features_base * 2, features_base * 4, (3,4,4), (1,2,2), 1), # 128x2x32x32
            self._block(features_base * 4, features_base * 8, 4, 2, 1), # 256x1x16x16
        )

        self.head = nn.Sequential(
            nn.Linear(features_base * 8, features_base * 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(features_base * 2, num_classes)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):

        return nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.1)
        )

    def _get_embedding(self, x):
        features = self.backbone(x)
        feature_embeds = torch.mean(features, dim=(-3, -2, -1))
        return feature_embeds

    def forward(self, x):
        embeds = self._get_embedding(x)
        return self.head(embeds)

class GeoFinderV4(nn.Module):

    def __init__(
        self,
        num_classes=42,
        model_id="openai/clip-vit-base-patch32",
        **kwargs
    ) -> None:
        super(GeoFinderV4, self).__init__(**kwargs)

        self.backbone = CLIPVisionModel.from_pretrained(model_id)

        self.head = nn.Linear(768, num_classes)

    def forward(self, x):

        views = torch.chunk(x["pixel_values"],chunks=2,dim=1)

        embeds = torch.stack(
            [self.backbone(pixel_values=view[:,0]).pooler_output for view in views],
            dim=-1
        ).mean(dim=-1)
        return self.head(embeds)

if __name__=="__main__":
    m = GeoFinderV1()
    m.eval()
    # inp = torch.randn((1,12,256,256),dtype=torch.float32)
    # m.enable_layer("layer4.2.conv1")  
    # output = m(inp)
    # print(output)
    # feats = m.intermediates["layer4.2.conv1"]
    # d = output[0][0]
    # print(torch.autograd.grad(d, feats, torch.ones(d.size()))[0].shape)