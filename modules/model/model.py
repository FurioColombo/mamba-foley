import torch
from torch import nn
from einops import rearrange, repeat

from types import SimpleNamespace

from modules.model.mamba_block import MambaBlock
from modules.model.model_layers import GBlock, Conv1d, RFF_MLP_Block, Residual, PreNorm, Attention, default, prob_mask_like

# --- U-Net ---
class UNet(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        print("Model initializing... This can take a few minutes.")

        # config file compatibility setup
        config_cond = config.condition if hasattr(config, 'condition') else config
        config_model = config.model if  hasattr(config, 'model') else config

        # Hyperparameter Settings
        sequential = config_model.sequential
        assert sequential in ['lstm', 'attn', 'mamba',
                              None], "Choose sequential between \'lstm\' or \'attn\' or \'mamba\', None."

        dims = config_model.dims
        factors = config_model.factors
        assert len(dims) - 1 == len(factors)

        block_nums = config_cond.block_nums

        time_emb_dim = config_cond.time_emb_dim
        class_emb_dim = config_cond.class_emb_dim
        event_dim = vars(config_cond.event_dims)[config_cond.event_type]
        cond_drop_prob = config_cond.cond_prob
        film_type = config_cond.film_type
        n_bottleneck_layers = config_model.bottleneck_layers if hasattr(config_model, 'bottleneck_layers') else 1
        bidirectional_bottleneck = config_model.bidirectional_bottleneck if hasattr(config_model, 'bidirectional_bottleneck') else True

        # Pre-conv/emb Layers
        self.conv_1 = Conv1d(1, dims[0], 5, padding=2)
        self.embedding = RFF_MLP_Block(time_emb_dim)

        # Up/DownSample Block Layers
        DBlock_list = []
        for in_dim, out_dim, factor, block_num in zip(dims[:-1], dims[1:], factors, block_nums):
            DBlock_list.append(GBlock(in_dim, out_dim, factor, block_num, film_type, event_dim))
        self.downsample = nn.ModuleList(DBlock_list)

        UBlock_list = []
        for in_dim, out_dim, factor, block_num in zip(dims[:0:-1], dims[-2::-1], factors[::-1], block_nums[::-1]):
            UBlock_list.append(GBlock(in_dim, out_dim, -1 * factor, block_num, film_type, event_dim))
        self.upsample = nn.ModuleList(UBlock_list)
        self.last_conv = Conv1d(dims[0], 1, 3, padding=1)

        # Bottleneck layer
        self.sequential = sequential
        if sequential:
            self.mid_dim = config_cond.mid_dim
            if sequential == 'lstm':
                self.lstm = nn.LSTM(input_size=self.mid_dim, hidden_size=self.mid_dim, num_layers=2, batch_first=True, bidirectional=True)
                self.lstm_mlp = nn.Sequential(
                    nn.Linear(self.mid_dim * 2, self.mid_dim),
                    nn.SiLU(),
                    nn.Linear(self.mid_dim, self.mid_dim)
                )
            if sequential == 'attn' or sequential == 'attention':
                self.mid_attn = Residual(PreNorm(self.mid_dim, Attention(self.mid_dim)))

            if sequential == 'mamba':
                self.bottleneck_mamba = MambaBlock(
                    in_channels=self.mid_dim,
                    n_layer=n_bottleneck_layers,
                    bidirectional=bidirectional_bottleneck
                )
                self.lstm_mlp = nn.Sequential(
                    nn.Linear(self.mid_dim * 2, self.mid_dim),
                    nn.SiLU(),
                    nn.Linear(self.mid_dim, self.mid_dim)
                )

        # Classifier-free guidance
        self.cond_drop_prob = cond_drop_prob

        self.classes_emb = nn.Embedding(num_classes, class_emb_dim)
        self.null_classes_emb = nn.Parameter(torch.randn(class_emb_dim))
        self.null_event_emb = nn.Parameter(torch.randn(event_dim))

        classes_dim = class_emb_dim * 4
        self.classes_mlp = nn.Sequential(
            nn.Linear(class_emb_dim, classes_dim),
            nn.SiLU(),
            nn.Linear(classes_dim, class_emb_dim)
        )
        print("Model successfully initialized!")

    def forward(self, audio, sigma, classes, events, cond_drop_prob=None):
        batch, device = audio.shape[0], audio.device
        x = audio.unsqueeze(1)
        x = self.conv_1(x)
        downsampled = []
        sigma_encoding = self.embedding(sigma)

        # Prepare Conditions(class, event)
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        classes_emb = self.classes_emb(classes)
        if cond_drop_prob[0] > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob[0], device=device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b=batch)

            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )
        c = self.classes_mlp(classes_emb)

        if cond_drop_prob[1] > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob[1], device=device)
            null_event = repeat(self.null_event_emb, 'd -> b d', b=batch)

            events = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                events,
                null_event
            ) if events != None else null_event

        # Downsample
        for layer in self.downsample:
            x = layer(x, sigma_encoding, c, events)
            downsampled.append(x)

        # Bottleneck
        if self.sequential:
            if self.sequential == 'lstm':
                h0 = torch.randn(4, batch, self.mid_dim, device=device)
                c0 = torch.randn(4, batch, self.mid_dim, device=device)
                x = x.permute(0, 2, 1)
                x, _ = self.lstm(x, (h0, c0))
                x = self.lstm_mlp(x)
                x = x.permute(0, 2, 1)

            if self.sequential == 'attn':
                x = self.mid_attn(x)

            if self.sequential == 'mamba':
                x = x.transpose(1,2)
                x = self.bottleneck_mamba(x)
                x = self.lstm_mlp(x)
                x = x.permute(0,2,1)

            x = x + downsampled[-1]  # residual connection

        # Upsample
        for layer, x_dblock in zip(self.upsample, reversed(downsampled)):
            x = torch.cat([x, x_dblock], dim=1)
            x = layer(x, sigma_encoding, c, events)

        x = self.last_conv(x)
        x = x.squeeze(1)
        return x

    def forward_with_cond_scale(self, audio, sigma, classes, event, cond_scale=1.):
        cond_score = self.forward(audio, sigma, classes, event, cond_drop_prob=[0.0, 0.0])
        if cond_scale == 1: return cond_score
        uncond_score = self.forward(audio, sigma, classes, event, cond_drop_prob=[1.0, 1.0])
        return uncond_score + (cond_score - uncond_score) * cond_scale


