"""Pooled normalized residual loss and its validation checks."""

import math
from numbers import Real

import torch
from torch import Tensor
from torch.nn import functional as F


class NoUsableCellsError(ValueError):
    """No cells satisfy both the input and observed-label conditions.

    A future training loop must explicitly skip such a batch and report its
    count; returning a numerical zero would disguise absent supervision.
    """


def _floating_tensor(value, name, shape, device):
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.device != device:
        raise ValueError(f"{name} must be on the prediction device")


def _sample_parameter(value, name, batch_size, device):
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if value.device != device:
        raise ValueError(f"{name} must be on the prediction device")
    if tuple(value.shape) not in ((batch_size,), (batch_size, 1)):
        raise ValueError(f"{name} must have shape [batch] or [batch, 1]")
    if not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
        raise ValueError(f"{name} must be finite and strictly positive")
    return value.reshape(batch_size, 1)


def _number(value, name, *, minimum, inclusive=True):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (not inclusive and value == minimum):
        comparison = "at least" if inclusive else "greater than"
        raise ValueError(f"{name} must be {comparison} {minimum}")


def normalized_residual_loss(
    predicted_residual: Tensor,
    target_residual: Tensor,
    *,
    input_ready: Tensor,
    label_loss_mask: Tensor,
    site_scale: Tensor,
    kind: str = "mse",
    huber_delta: float = 1.0,
    observed_discharge: Tensor | None = None,
    peak_threshold: Tensor | None = None,
    peak_weight: float = 2.0,
    peak_cap: float = 5.0,
) -> Tensor:
    """Return a scalar objective for physical-unit residual predictions.

    ``predicted_residual`` and ``target_residual`` have shape ``[batch, lead]``.
    ``input_ready`` and ``label_loss_mask`` are independent Boolean tensors of
    that shape; only their intersection enters the loss. A missing verification
    label never decides input readiness or forecast issuance. Invalid values in
    unselected cells are ignored by indexing *before* numerical operations.
    Selected predictions and labels must be finite.

    ``site_scale`` is a strictly positive, finite fit-only site statistic,
    expanded to each sample as ``[batch]`` or ``[batch, 1]``. MSE is the selected
    cell mean of ``((prediction - target) / scale)**2``. Huber uses the same
    normalized error and a transition at ``huber_delta`` normalized units.

    ``peak_mse`` additionally requires observed discharge ``[batch, lead]`` and
    a positive fit-only discharge threshold per sample/site. Its weight is
    ``min(peak_cap, 1 + peak_weight * max(discharge / threshold - 1, 0))``;
    the loss is the weighted sum of squared normalized errors divided by the
    sum of weights. Neither discharge labels nor peak weights may enter a
    model's forward inputs. Changing the threshold or weighting requires a
    fit/validation decision before any evaluation access.

    Raises ``NoUsableCellsError`` for an empty intersection. Callers should skip
    and count such batches rather than run an optimizer step on absent labels.
    """
    if not isinstance(predicted_residual, Tensor) or not predicted_residual.is_floating_point():
        raise ValueError("predicted_residual must be a floating-point tensor")
    if predicted_residual.ndim != 2 or 0 in predicted_residual.shape:
        raise ValueError("predicted_residual must have nonempty shape [batch, lead]")
    shape = tuple(predicted_residual.shape)
    device = predicted_residual.device
    _floating_tensor(target_residual, "target_residual", shape, device)
    for mask, name in ((input_ready, "input_ready"), (label_loss_mask, "label_loss_mask")):
        if not isinstance(mask, Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != shape:
            raise ValueError(f"{name} must be a Boolean tensor with shape {shape}")
        if mask.device != device:
            raise ValueError(f"{name} must be on the prediction device")
    if kind not in ("mse", "huber", "peak_mse"):
        raise ValueError("kind must be mse, huber or peak_mse")
    scale = _sample_parameter(site_scale, "site_scale", shape[0], device)
    if kind == "huber":
        _number(huber_delta, "huber_delta", minimum=0.0, inclusive=False)
    if kind == "peak_mse":
        _number(peak_weight, "peak_weight", minimum=0.0)
        _number(peak_cap, "peak_cap", minimum=1.0)
        _floating_tensor(observed_discharge, "observed_discharge", shape, device)
        threshold = _sample_parameter(peak_threshold, "peak_threshold", shape[0], device)

    selected = input_ready & label_loss_mask
    if not bool(selected.any()):
        raise NoUsableCellsError("no cells have both ready inputs and usable observed labels")
    prediction = predicted_residual[selected]
    target = target_residual[selected]
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("selected residual predictions and targets must be finite")
    normalized_error = (prediction - target) / scale.expand(shape)[selected]
    if not bool(torch.isfinite(normalized_error).all()):
        raise ValueError("normalized residual errors overflowed; check units and site scales")
    if kind == "huber":
        result = F.huber_loss(normalized_error, torch.zeros_like(normalized_error),
                              reduction="mean", delta=huber_delta)
    elif kind == "peak_mse":
        discharge = observed_discharge[selected]
        if not bool(torch.isfinite(discharge).all()) or not bool((discharge >= 0).all()):
            raise ValueError("selected observed discharge must be finite and nonnegative")
        relative_peak = discharge / threshold.expand(shape)[selected]
        weights = (1.0 + peak_weight * (relative_peak - 1.0).clamp_min(0.0)).clamp_max(peak_cap)
        result = (weights * normalized_error.square()).sum() / weights.sum()
    else:
        result = normalized_error.square().mean()
    if not bool(torch.isfinite(result)):
        raise ValueError("loss overflowed; check residual units and site scales")
    return result
