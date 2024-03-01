import numpy as np
import tinycudann as tcnn
import torch
import torch.nn as nn
from activation import trunc_exp

from .multi_object_renderer import MultiObjectNeRFRenderer


# Load a pretrained YOLOv8n model


class MultiModelList(nn.ModuleList):
    def __init__(self, *inputs):
        super().__init__(*inputs)
        pass

    def forward(self, *inputs, **kwargs):
        return torch.stack([module(*inputs, **kwargs) for module in self])
        # return torch.stack([module(*inputs, **kwargs) for module in self])


class MultObjectNerf(MultiObjectNeRFRenderer):
    def __init__(self,
                 encoding="HashGrid",
                 encoding_dir="SphericalHarmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,

                 bound=1,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))
        self.average_object_features = None
        self.semantic_objects = []
        _encoder, _encoder_dir, _sigma_net, _color_net = self.init_submodule()
        self.encoder = MultiModelList([_encoder])
        self.encoder_dir = MultiModelList([_encoder_dir])
        self.sigma_net = MultiModelList([_sigma_net])
        self.color_net = MultiModelList([_color_net])
        self.optimizer=None
        # self.yolo = SemanticExtractor()
        # self.model = NeRFNetwork(*args, **kwargs)

    def add_new_head(self, label):
        if not label in self.semantic_objects:
            print("Adding New Head")
            self.semantic_objects.append(label)
            _encoder, _encoder_dir, _sigma_net, _color_net = self.init_submodule()
            self.encoder.append(_encoder)
            self.encoder_dir.append(_encoder_dir)
            self.sigma_net.append(_sigma_net)
            self.color_net.append(_color_net)
            self.optimizer(self)

    def init_submodule(self):
        encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": self.per_level_scale,
            },
        )

        sigma_net = tcnn.Network(
            n_input_dims=32,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.hidden_dim,
                "n_hidden_layers": self.num_layers - 1,
            },
        )

        encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        in_dim_color = encoder_dir.n_output_dims + self.geo_feat_dim

        color_net = tcnn.Network(
            n_input_dims=in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.hidden_dim_color,
                "n_hidden_layers": self.num_layers_color - 1,
            },
        )
        return encoder, encoder_dir, sigma_net, color_net

    def density(self, x, label=None, ):
        # x: [N, 3], in [-bound, bound]
        if label is not None:
            self.add_new_head(label)
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        if label in self.semantic_objects:
            idx = self.semantic_objects.index(label)
            x = self.encoder[idx](x)
            h = self.sigma_net[idx](x)
        elif label == 'all':
            x = self.encoder(x)
            h = self.sigma_net(x)
        else:
            # breakpoint()
            x = self.encoder[0](x)
            h = self.sigma_net[0](x)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]
        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    # allow masked inference
    def color(self, x, d, label=None, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.
        if label is not None:
            self.add_new_head(label)
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device)  # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        # color
        d = (d + 1) / 2  # tcnn SH encoding requires inputs to be in [0, 1]
        if label in self.semantic_objects:
            idx = self.semantic_objects.index(label)
            d = self.encoder_dir[idx](d)
        elif label == 'all':
            d = self.encoder_dir(d)
        else:
            d = self.encoder_dir[0](d)

        h = torch.cat([d, geo_feat], dim=-1)

        if label in self.semantic_objects:
            idx = self.semantic_objects.index(label)
            h = self.color_net[idx](h)
        elif label == 'all':
            h = self.color_net(h)
        else:
            h = self.color_net[0](h)
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)  # fp16 --> fp32
        else:
            rgbs = h
        return rgbs
    def set_optimizer(self, optimizer):
        self.optimizer =optimizer
    def forward(self, x, d, label=None):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # sigma
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        if label is not None:
            if not label in self.semantic_objects:
                self.semantic_objects.append(label)
        if label in self.semantic_objects:
            idx = self.semantic_objects.index(label)
            x = self.encoder[idx](x)
            h = self.sigma_net[idx](x)
        elif label == 'all':
            x = self.encoder(x)
            h = self.sigma_net(x)
        else:
            x = self.encoder[0](x)
            h = self.sigma_net[0](x)
        # attempt merge of densities
        if label == "all":
            breakpoint()
        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])

        geo_feat = h[..., 1:]

        # color
        d = (d + 1) / 2  # tcnn SH encoding requires inputs to be in [0, 1]
        if label in self.semantic_objects:
            idx = self.semantic_objects.index(label)
            d = self.encoder_dir[idx](d)
        elif label == 'all':
            d = self.encoder_dir(d)
        else:
            d = self.encoder_dir[0](d)

        h = torch.cat([d, geo_feat], dim=-1)

        if label in self.semantic_objects:
            idx = self.semantic_objects.index(label)
            h = self.color_net[idx](h)
        elif label == 'all':
            h = self.color_net(h)
        else:
            h = self.color_net[0](h)

        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color

    # optimizer utils
    def get_params(self, lr):
        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        return params
