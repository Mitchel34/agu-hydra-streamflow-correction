"""Model configuration for the three site-specific native H1–H18 model families."""

from dataclasses import asdict, dataclass, fields
import math


CANDIDATES = ("seq2seq_lstm", "hydra_gru_conv", "temporal_transformer")
NATIVE_TIMING_POLICY = "native_issue_1_18_no_acquisition_offset"


@dataclass(frozen=True)
class ModelConfig:
    site_id: str
    candidate: str = "seq2seq_lstm"
    history_length: int = 168
    history_patch_size: int = 3
    history_channels: tuple[str, ...] = ("usgs:discharge",)
    future_channels: tuple[str, ...] = (
        "nwm:discharge", "hrrr:precipitation_1h", "hrrr:temperature_2m",
        "hrrr:specific_humidity_2m", "hrrr:snow_water_equivalent",
    )
    initial_state_channels: tuple[str, ...] = ("hrrr:soil_moisture_initial",)
    decision_leads: tuple[int, ...] = tuple(range(1, 19))
    timing_policy: str = NATIVE_TIMING_POLICY
    hidden_size: int = 64
    recurrent_layers: int = 1
    attention_layers: int = 2
    num_heads: int = 4
    feedforward_multiplier: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        if not isinstance(self.site_id, str) or not self.site_id.strip():
            raise ValueError("an explicit single site_id is required")
        if self.candidate not in CANDIDATES:
            raise ValueError("unknown model candidate")
        if self.timing_policy != NATIVE_TIMING_POLICY:
            raise ValueError("unknown model timing policy")
        for name in ("history_length", "history_patch_size", "hidden_size", "recurrent_layers", "attention_layers",
                     "num_heads", "feedforward_multiplier"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.candidate == "temporal_transformer" and self.history_length % self.history_patch_size:
            raise ValueError("temporal history_length must be divisible by history_patch_size")
        for name in ("history_channels", "future_channels", "initial_state_channels", "decision_leads"):
            value = getattr(self, name)
            if isinstance(value, (str, bytes)):
                raise ValueError(f"{name} must be an ordered collection")
            object.__setattr__(self, name, tuple(value))
            if not value or len(set(value)) != len(value):
                raise ValueError(f"{name} must be nonempty and unique")
        channels = self.history_channels + self.future_channels + self.initial_state_channels
        if any(not isinstance(c, str) or not c.strip() for c in channels) or len(set(channels)) != len(channels):
            raise ValueError("channel names must be unique nonempty strings across axes")
        if any(isinstance(h, bool) or not isinstance(h, int) or not 1 <= h <= 18 for h in self.decision_leads):
            raise ValueError("decision leads must be positive integer hours <=18")
        if tuple(sorted(self.decision_leads)) != self.decision_leads:
            raise ValueError("decision leads must be increasing")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)) or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite in [0,1)")

    @property
    def horizon(self):
        return len(self.decision_leads)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, body):
        if not isinstance(body, dict) or set(body) - {f.name for f in fields(cls)}:
            raise ValueError("unknown model configuration fields")
        return cls(**body)
