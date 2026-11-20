"""Full-input neural architectures used in the native H1–H18 study."""

import torch
from torch import nn

from .config import ModelConfig
from .inputs import ModelInputs


class _Candidate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def _check(self, inputs):
        if not isinstance(inputs, ModelInputs):
            raise TypeError("forward accepts only predictor-only ModelInputs")
        inputs.validate(self.config)


def _lead_features(inputs, config):
    leads = inputs.future.values.new_tensor(config.decision_leads) / 18.0
    return leads[None, :, None].expand(inputs.history.values.shape[0], -1, -1)


class Seq2SeqLSTM(_Candidate):
    """USGS time encoder; sequential known-forecast decoder; linear residual."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        d, layers = config.hidden_size, config.recurrent_layers
        self.context = nn.Sequential(nn.Linear(5 * len(config.initial_state_channels), d), nn.GELU())
        self.history_projection = nn.Linear(5 * len(config.history_channels) + 1, d)
        self.future_projection = nn.Linear(6 * len(config.future_channels) + 1 + d, d)
        recurrent_dropout = config.dropout if layers > 1 else 0.0
        self.encoder = nn.LSTM(d, d, layers, batch_first=True, dropout=recurrent_dropout)
        self.decoder = nn.LSTM(d, d, layers, batch_first=True, dropout=recurrent_dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(d, 1)

    def forward(self, inputs: ModelInputs):
        self._check(inputs)
        c = self.config
        context = self.context(inputs.initial_state.packed().flatten(1))
        history = inputs.history.packed().flatten(2)
        offsets = torch.arange(1 - c.history_length, 1, device=history.device, dtype=history.dtype) / c.history_length
        time = offsets[None, :, None].expand(history.shape[0], -1, -1)
        sequence = self.dropout(torch.nn.functional.gelu(self.history_projection(torch.cat((history, time), dim=-1))))
        initial = context.unsqueeze(0).repeat(c.recurrent_layers, 1, 1).contiguous()
        _, state = self.encoder(sequence, (initial, torch.zeros_like(initial)))
        future = torch.cat((inputs.future_packed().flatten(2), _lead_features(inputs, c),
                            context[:, None, :].expand(-1, c.horizon, -1)), dim=-1)
        decoded, _ = self.decoder(self.dropout(torch.nn.functional.gelu(self.future_projection(future))), state)
        return self.head(self.dropout(decoded)).squeeze(-1)


class HydraGRUConv(_Candidate):
    """Chronological history GRU plus origin-known, noncausal forecast CNN.

    Soil state is encoded once as context, never fabricated as a time series.
    The same residual head is applied at each native forecast lead.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        d = config.hidden_size
        self.context = nn.Sequential(nn.Linear(5 * len(config.initial_state_channels), d), nn.GELU())
        self.history_projection = nn.Linear(5 * len(config.history_channels) + 1, d)
        self.encoder = nn.GRU(d, d, config.recurrent_layers, batch_first=True,
            dropout=config.dropout if config.recurrent_layers > 1 else 0.0)
        self.future_projection = nn.Linear(6 * len(config.future_channels) + 1, d)
        self.forecast_conv = nn.Sequential(
            nn.Conv1d(d, d, kernel_size=3, padding=1), nn.GELU(), nn.Dropout(config.dropout),
            nn.Conv1d(d, d, kernel_size=3, padding=1), nn.GELU())
        self.head = nn.Sequential(nn.Linear(3 * d + 1, d), nn.GELU(),
                                  nn.Dropout(config.dropout), nn.Linear(d, 1))

    def forward(self, inputs: ModelInputs):
        self._check(inputs)
        c = self.config
        context = self.context(inputs.initial_state.packed().flatten(1))
        history = inputs.history.packed().flatten(2)
        offsets = torch.arange(1 - c.history_length, 1, device=history.device,
                               dtype=history.dtype) / c.history_length
        time = offsets[None, :, None].expand(history.shape[0], -1, -1)
        sequence = torch.nn.functional.gelu(self.history_projection(torch.cat((history, time), -1)))
        _, encoded = self.encoder(sequence, context[None].repeat(c.recurrent_layers, 1, 1).contiguous())
        lead = _lead_features(inputs, c)
        forecast = self.future_projection(torch.cat((inputs.future_packed().flatten(2), lead), -1))
        forecast = self.forecast_conv(forecast.transpose(1, 2)).transpose(1, 2)
        fused = torch.cat((forecast, encoded[-1, :, None].expand(-1, c.horizon, -1),
                           context[:, None].expand(-1, c.horizon, -1), lead), -1)
        return self.head(fused).squeeze(-1)


def _initialize_attention_layers(layers):
    # PyTorch stacks clone a prototype: initialize projections independently.
    for block in layers:
        for module in block.modules():
            if isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.zeros_(module.in_proj_bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class TemporalTransformer(_Candidate):
    """Ordered history patches and known-forecast queries, never target tokens.

    Missing patches/forecast rows cannot act as attention keys. Always-present
    learned context tokens keep all-missing rows finite; numerical outputs do
    not authorize issuance. Decoder attention is noncausal because every input
    forecast is already known at the origin, including later forecast leads.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        d = config.hidden_size
        patches = config.history_length // config.history_patch_size
        self.history_projection = nn.Linear(config.history_patch_size * 5 * len(config.history_channels), d)
        self.history_positions = nn.Embedding(patches, d)
        self.future_projection = nn.Linear(6 * len(config.future_channels) + 1, d)
        self.lead_positions = nn.Embedding(19, d)
        self.state_projection = nn.Linear(5 * len(config.initial_state_channels), d)
        self.history_context = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.query_context = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d, config.num_heads,
            dim_feedforward=d * config.feedforward_multiplier, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.attention_layers,
            norm=nn.LayerNorm(d), enable_nested_tensor=False)
        decoder_layer = nn.TransformerDecoderLayer(d, config.num_heads,
            dim_feedforward=d * config.feedforward_multiplier, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, config.attention_layers, norm=nn.LayerNorm(d))
        _initialize_attention_layers(list(self.encoder.layers) + list(self.decoder.layers))
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                  nn.Dropout(config.dropout), nn.Linear(d, 1))

    def forward(self, inputs: ModelInputs):
        self._check(inputs)
        c = self.config
        batch = inputs.history.values.shape[0]
        patches = c.history_length // c.history_patch_size
        packed = inputs.history.packed().reshape(batch, patches, -1)
        history = self.history_projection(packed)
        history = history + self.history_positions(torch.arange(patches, device=history.device))[None]
        always_present = torch.zeros((batch, 1), dtype=torch.bool, device=history.device)
        history_padding = ~inputs.history.usable.reshape(batch, patches, -1).any(-1)
        history_padding = torch.cat((always_present, history_padding), 1)
        history = torch.cat((self.history_context.expand(batch, -1, -1), history), 1)
        memory = self.encoder(history, src_key_padding_mask=history_padding)
        state = self.state_projection(inputs.initial_state.packed().flatten(1))[:, None]
        memory = torch.cat((memory, state), 1)
        memory_padding = torch.cat((history_padding, ~inputs.initial_state.usable.any(-1, keepdim=True)), 1)
        lead = _lead_features(inputs, c)
        queries = self.future_projection(torch.cat((inputs.future_packed().flatten(2), lead), -1))
        lead_indices = torch.tensor(c.decision_leads, dtype=torch.long, device=queries.device)
        queries = queries + self.lead_positions(lead_indices)[None]
        queries = torch.cat((self.query_context.expand(batch, -1, -1), queries), 1)
        query_padding = torch.cat((always_present, ~inputs.future.usable.any(-1)), 1)
        decoded = self.decoder(queries, memory, tgt_key_padding_mask=query_padding,
                               memory_key_padding_mask=memory_padding)
        return self.head(decoded[:, 1:]).squeeze(-1)


def build_model(config: ModelConfig):
    if not isinstance(config, ModelConfig):
        raise TypeError("build_model requires ModelConfig")
    families = {
        "seq2seq_lstm": Seq2SeqLSTM,
        "hydra_gru_conv": HydraGRUConv,
        "temporal_transformer": TemporalTransformer,
    }
    return families[config.candidate](config)
