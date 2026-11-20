"""Predictor-only tensors. Labels and reconstruction baselines stay separate.

The upstream loader owns UTC/source/split admission. These tensor checks cannot
prove historical availability or a truthful source vintage. There is no file
loader or authorization bypass in this module.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .config import ModelConfig, NATIVE_TIMING_POLICY


@dataclass(frozen=True)
class MaskedFeatures:
    values: Tensor
    usable: Tensor
    observed: Tensor
    imputed: Tensor
    age_hours: Tensor

    def packed(self):
        """Use neutral tensor placeholders with visible masks, not data repair.

        Features must already use frozen fit-only transforms. Age means donor
        age for history and issue-to-origin age for forecasts/initial state.
        Masked raw values/ages never enter the network. An output on an
        ineligible input is not an authorized corrected forecast.
        """
        values = torch.where(self.usable, self.values, 0.0)
        ages = torch.where(self.usable, self.age_hours, 0.0)
        return torch.stack((values, self.usable.to(values.dtype), self.observed.to(values.dtype),
                            self.imputed.to(values.dtype), torch.log1p(ages) / math.log(169)), dim=-1)


@dataclass(frozen=True)
class ModelInputs:
    history: MaskedFeatures
    future: MaskedFeatures
    initial_state: MaskedFeatures
    future_native_lead_hours: Tensor
    site_id: str

    def validate(self, config: ModelConfig):
        if not isinstance(self.site_id, str) or self.site_id != config.site_id:
            raise ValueError("input site_id must match this independently trained model's site")
        if not isinstance(self.history, MaskedFeatures):
            raise ValueError("history must be MaskedFeatures")
        reference = self.history.values
        if not isinstance(reference, Tensor) or not reference.is_floating_point():
            raise ValueError("history values must be floating tensors")
        if reference.ndim != 3 or reference.shape[0] == 0:
            raise ValueError("history must have a nonempty [batch,time,channel] axis")
        batch = reference.shape[0]
        shapes = ((batch, config.history_length, len(config.history_channels)),
                  (batch, config.horizon, len(config.future_channels)),
                  (batch, len(config.initial_state_channels)))
        for name, group, shape in zip(("history", "future", "initial_state"),
                                     (self.history, self.future, self.initial_state), shapes):
            if not isinstance(group, MaskedFeatures):
                raise ValueError(f"{name} must be MaskedFeatures")
            for field in ("values", "usable", "observed", "imputed", "age_hours"):
                tensor = getattr(group, field)
                if not isinstance(tensor, Tensor) or tuple(tensor.shape) != shape:
                    raise ValueError(f"{name}.{field} shape must be {shape}")
                if tensor.device != reference.device:
                    raise ValueError("all input tensors must share device")
                expected = torch.bool if field in ("usable", "observed", "imputed") else reference.dtype
                if tensor.dtype != expected:
                    raise ValueError(f"{name}.{field} has wrong dtype")
            if bool((group.observed & group.imputed).any()):
                raise ValueError("observed and imputed masks must not overlap")
            if bool((group.usable & ~(group.observed | group.imputed)).any()):
                raise ValueError("usable values require original or imputed lineage")
            if name != "history" and bool(group.imputed.any()):
                raise ValueError("forecast and initial-state imputation is unadmitted")
            if not bool(torch.isfinite(group.values[group.usable]).all()):
                raise ValueError("usable feature values must be finite")
            ages = group.age_hours[group.usable]
            if not bool((torch.isfinite(ages) & (ages >= 0)).all()):
                raise ValueError("usable feature ages must be finite and nonnegative")
        native = self.future_native_lead_hours
        if not isinstance(native, Tensor) or tuple(native.shape) != shapes[1] or native.dtype != reference.dtype or native.device != reference.device:
            raise ValueError("future_native_lead_hours must match future values shape/dtype/device")
        leads = native.new_tensor(config.decision_leads)[None, :, None]
        expected = leads + self.future.age_hours
        mask = self.future.usable
        if not bool((torch.isfinite(native[mask]) & (native[mask] > 0)).all()) or not torch.allclose(native[mask], expected[mask]):
            raise ValueError("native lead must equal decision lead plus issue age on usable forecasts")
        for index, channel in enumerate(config.future_channels):
            selected = mask[:, :, index]
            if channel == "nwm:discharge":
                if bool((native[:, :, index][selected] > 18).any()):
                    raise ValueError("NWM native lead exceeds H18")
                if config.timing_policy == NATIVE_TIMING_POLICY and bool((self.future.age_hours[:, :, index][selected] != 0).any()):
                    raise ValueError("native NWM issue must equal origin; acquisition offset is zero")
            elif channel.startswith("hrrr:") and bool((native[:, :, index][selected] > 36).any()):
                raise ValueError("HRRR native lead exceeds supplied H36 support")

    def future_packed(self):
        native = torch.where(self.future.usable, self.future_native_lead_hours, 0.0) / 18.0
        return torch.cat((self.future.packed(), native.unsqueeze(-1)), dim=-1)
