"""Fixed-capacity input ablations on the unchanged full-input cohort.

Apply this wrapper AFTER the existing full archive validation, eligibility,
historical-gap policy and fit-only standardization. Withheld channels become
constant zero values with false usability/observed/imputed masks, zero ages
and zero native forecast leads. The network cannot receive source information
through those fields. Channel slots and learned parameter counts stay fixed.

This estimates incremental predictive value conditional on full-input support.
It does not measure the maximal coverage obtainable without a source, causal
importance, or the performance of a smaller architecture. Positional encodings
and the site identifier remain. In ``no_nwm_predictor``, NWM still supplies the
residual target and additive reconstruction baseline outside the forward API.
No labels, baseline tensors, raw source references or readiness masks enter
this module, and it never changes admission or opens a data file.
"""

from dataclasses import dataclass, replace
import hashlib
import json

import torch
from torch import nn

from ..models import ModelConfig, ModelInputs, MaskedFeatures, build_model


_HISTORY = ("usgs:discharge",)
_FUTURE = (
    "nwm:discharge", "hrrr:precipitation_1h", "hrrr:temperature_2m",
    "hrrr:specific_humidity_2m", "hrrr:snow_water_equivalent",
)
_STATE = ("hrrr:soil_moisture_initial",)
_WITHHELD = {
    "full": (),
    "no_usgs": _HISTORY,
    "no_hrrr": _FUTURE[1:] + _STATE,
    "no_nwm_predictor": (_FUTURE[0],),
    "no_precipitation": (_FUTURE[1],),
    "no_temperature": (_FUTURE[2],),
    "no_humidity": (_FUTURE[3],),
    "no_snow": (_FUTURE[4],),
    "no_soil": _STATE,
    "nwm_only_predictors": _HISTORY + _FUTURE[1:] + _STATE,
}
ABLATION_NAMES = tuple(_WITHHELD)


@dataclass(frozen=True)
class AblationSpec:
    """Named, immutable projection recipe independent of outcomes or labels."""

    variant: str

    def __post_init__(self):
        if not isinstance(self.variant, str) or self.variant not in _WITHHELD:
            raise ValueError(f"unknown ablation variant: {self.variant!r}")

    @property
    def withheld_channels(self):
        return _WITHHELD[self.variant]

    def to_dict(self):
        return {
            "schema": "agu-hydra-fixed-capacity-ablation/v1",
            "variant": self.variant,
            "withheld_channels": list(self.withheld_channels),
            "input_axes": {"history": list(_HISTORY), "future": list(_FUTURE),
                           "initial_state": list(_STATE)},
            "withheld_fields": ["values", "usable", "observed", "imputed",
                                "age_hours", "future_native_lead_hours"],
            "projection_stage": "after_frozen_fit_only_standardization",
            "replacement": "constant_zero_values_ages_leads_and_false_masks",
            "cohort_policy": "unchanged_full_input_matched_cohort",
            "capacity_policy": "unchanged_channel_slots_and_parameter_count",
            "nwm_residual_anchor_retained": True,
            "source_provenance": "retained_externally_never_predictor_input",
        }

    @property
    def sha256(self):
        encoded = json.dumps(self.to_dict(), sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()


def _check_config(config):
    if not isinstance(config, ModelConfig):
        raise TypeError("ablation projection requires ModelConfig")
    if (config.history_channels != _HISTORY or config.future_channels != _FUTURE
            or config.initial_state_channels != _STATE):
        raise ValueError("fixed-capacity ablations require the complete named input axes")


def _neutralize(group, channels, withheld):
    indices = [i for i, channel in enumerate(channels) if channel in withheld]
    if not indices:
        return group
    # Use new tensors: original archive tensors and cached standardized batches
    # may be replayed by other variants and must remain unchanged.
    fields = {}
    for name in ("values", "usable", "observed", "imputed", "age_hours"):
        value = getattr(group, name).clone()
        value[..., indices] = 0
        fields[name] = value
    return MaskedFeatures(**fields)


def project_inputs(inputs, config, spec):
    """Validate, then hide the selected predictor fields without changing rows.

    The existing full-input validation still applies even to withheld fields.
    Invalid source tensors cannot be admitted by hiding their errors. Callers
    retain targets, NWM reconstruction baselines and matched-cohort masks.
    """
    _check_config(config)
    if not isinstance(spec, AblationSpec):
        raise TypeError("spec must be an AblationSpec")
    if not isinstance(inputs, ModelInputs):
        raise TypeError("ablation forward accepts predictor-only ModelInputs")
    inputs.validate(config)
    if not spec.withheld_channels:
        return inputs
    withheld = frozenset(spec.withheld_channels)
    future_indices = [i for i, name in enumerate(config.future_channels)
                      if name in withheld]
    native = inputs.future_native_lead_hours
    if future_indices:
        native = native.clone()
        native[..., future_indices] = 0
    projected = replace(
        inputs,
        history=_neutralize(inputs.history, config.history_channels, withheld),
        future=_neutralize(inputs.future, config.future_channels, withheld),
        initial_state=_neutralize(inputs.initial_state, config.initial_state_channels, withheld),
        future_native_lead_hours=native,
    )
    projected.validate(config)
    return projected


class AblationModel(nn.Module):
    """Predictor-only wrapper compatible with the existing training engine.

    The wrapper's state dictionary includes a primitive-only extra-state
    recipe/hash. Loading a checkpoint through a different ablation fails,
    including with ``strict=False``. Recreate this wrapper with the recorded
    variant before loading its state dictionary. Older unwrapped checkpoints
    are intentionally not treated as interchangeable ablation checkpoints.
    """

    def __init__(self, config, variant="full"):
        super().__init__()
        _check_config(config)
        self.config = config
        self.ablation_spec = AblationSpec(variant)
        self.base_model = build_model(config)

    def ablation_provenance(self):
        return {"specification": self.ablation_spec.to_dict(),
                "specification_sha256": self.ablation_spec.sha256}

    def get_extra_state(self):
        return self.ablation_provenance()

    def set_extra_state(self, state):
        if state != self.ablation_provenance():
            raise ValueError("checkpoint ablation specification/hash does not match this model")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Verify before the base loader can alter any learned weights, and
        # refuse unbound checkpoints even if the caller relaxes key checking.
        if state_dict.get("_extra_state") != self.ablation_provenance():
            raise ValueError("checkpoint lacks the matching ablation specification/hash")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, inputs):
        return self.base_model(project_inputs(inputs, self.config, self.ablation_spec))


def build_ablation_model(config, variant="full"):
    """Seed before construction, as for ``build_model``. No execution occurs."""
    return AblationModel(config, variant)
