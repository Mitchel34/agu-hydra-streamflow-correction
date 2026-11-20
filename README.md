# HYDRA · Local correction of National Water Model streamflow forecasts

**AGU26 poster H51P-1404** · Mitchel Carson, Mohammad Ali Javidian, and William P. Anderson Jr.

Can recent streamflow and local weather forecasts improve the National Water Model at specific gauges? This study trains three neural architectures to predict an additive correction to each of 18 hourly NWM forecasts. It then retrains each architecture with one input group removed to measure what that group contributed.

[Explore the study and interactive figures](https://www.hydramodel.ai/) · [Read the methods and reproduce the score tables](METHODS.md) · [Full technical results](https://www.hydramodel.ai/technical)

## Results

Across the LSTM, GRU with convolutions, and temporal Transformer, pooled squared error relative to NWM was **41–46% lower at Jefferson** and **36–49% lower at Galax** in the evaluation year. At Sugar Grove the three architectures finished within about 1% of NWM. Retraining without initial HRRR soil moisture raised pooled squared error by 16–30% at Jefferson and 22–61% at Galax.

At Jefferson all three architectures reduced both mean absolute error and the standard deviation of absolute error from hours 2 through 18. At 18 hours, MAE fell by 30–38% and its casewise spread by 15–20%. The three models were fitted separately at each gauge. [By-lead scores](results/scores_by_lead.csv) · [Jefferson absolute-error spread](results/jefferson_absolute_sd_by_lead.csv) · [Input-removal results](results/input_removals.csv)

## Explore the code

- [Network architectures](src/agu_hydra/models/architectures.py) include a sequence-to-sequence LSTM, a GRU with forecast convolutions, and a temporal Transformer. Each produces a signed residual for each lead.
- [Predictor tensors](src/agu_hydra/models/inputs.py) carry the data, masks, ages, native leads and gauge identity.
- [Training loss](src/agu_hydra/models/losses.py) applies the pooled normalized MSE used in the study to forecast-ready cells with observed verification.
- [Fixed-capacity input removals](src/agu_hydra/training/ablations.py) preserve model capacity and the matched forecast cases while withholding each designated input group.
- [Score-table exporter](scripts/export_scorecards.py) regenerates the two CSV tables using Python's standard library and the pinned published aggregate release.

## Citation

Carson, M., M. A. Javidian, and W. P. Anderson Jr. (2026). *Hydra: Identifying Data That Improve Local National Water Model Streamflow Estimates*. AGU Annual Meeting, poster H51P-1404. [Citation metadata](CITATION.cff)

Model selection reused development validation, and an earlier project had evaluated the same 2022–23 period. The results describe retrospective skill on the eligible observed cases at these three gauges. [Study findings and limitations](https://www.hydramodel.ai/files/FINDINGS.md)
