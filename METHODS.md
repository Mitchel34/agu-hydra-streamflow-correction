# Methods and reproducibility

## Forecasts and observations

At each gauge, the model consumes 168 hourly gauge-history slots up to the forecast origin, a sequence of NWM and HRRR predictors, and HRRR initial soil moisture. HRRR supplies precipitation, 2 m temperature, 2 m specific humidity and snow water equivalent. Each sample retains the native NWM issue and valid times for leads 1 through 18. The verification target is observed USGS discharge at the same gauge and valid hour. Eligible gauge history may hold an original observation for up to two hours under the frozen rules. Missing verification labels remain missing.

| USGS station | Gauge |
|---|---|
| [03161000](https://waterdata.usgs.gov/monitoring-location/USGS-03161000/) | South Fork New River near Jefferson, North Carolina |
| [03164000](https://waterdata.usgs.gov/monitoring-location/USGS-03164000/) | New River near Galax, Virginia |
| [03479000](https://waterdata.usgs.gov/monitoring-location/USGS-03479000/) | Watauga River near Sugar Grove, North Carolina |

The campaign used archived USGS discharge records, previously exported NWM H1–H18 short-range streamflow, and HRRR weather and soil inputs. The archived inputs have incomplete issue-time availability histories. The public acquisition record identifies the source products, processing, and checksums. Exact regeneration of the original training panels requires the referenced exports. [Archive and panel provenance](https://github.com/Mitchel34/hydra-nwm-streamflow-correction/blob/3826e5d78389b58be04cf7e773fbe05fc04ac920/dashboard/agu-hrrr/SOURCES.md#archive-inputs-and-panel-construction)

## Training and comparison

Each model belongs to one gauge. The target in physical units is observed flow minus NWM. At scoring, the rescaled predicted residual is added to the native NWM value. The evaluated forecasts retain the signed sum, including negative predictions. Scaling uses only the fitting period for development and only the first year for the final first-year refits.

Development fitted May 2021 through January 2022 and selected settings on February through April 2022. All three families used chronological batches of 32, AdamW with weight decay 0.0001, unit gradient-norm clipping and pooled normalized MSE. The earliest trained epoch with the lowest validation loss fixed the training length for its paired first-year refit. Patience was 10. Each family shared its settings across the three independently trained site models.

| Family | Fixed learning rate | Development epoch cap |
|---|---:|---:|
| Sequence-to-sequence LSTM | 0.0003 | 200 |
| GRU with forecast convolutions | 0.0001 | 100 |
| Temporal Transformer | 0.0003 | 100 |

Each site and family used seeds 1729, 1730 and 1731. The full inputs and eight removals yielded 243 development fits, followed by 243 first-year refits scored from May 2022 through April 2023. Removing NWM as a predictor retains NWM as the final forecast anchor. The complete evaluation includes simple local corrections. An earlier project had already scored this evaluation year and informed study design. Storms are shared across gauges, so these comparisons do not establish independent flood-event skill. [Complete scorecards and evaluation report](https://www.hydramodel.ai/files/REPORT.md)

## Reproduce the published tables

Use Python 3.11 or later. From the repository root, obtain the public, hash-verified aggregate release and write the tables to a new directory.

```sh
curl --fail --location --output data.json https://raw.githubusercontent.com/Mitchel34/hydra-nwm-streamflow-correction/3826e5d78389b58be04cf7e773fbe05fc04ac920/dashboard/agu-hrrr/site/data.json
python3 scripts/export_scorecards.py data.json reproduced
diff results/scores_by_lead.csv reproduced/scores_by_lead.csv
diff results/input_removals.csv reproduced/input_removals.csv
```

The lead scorecard contains 216 rows covering three gauges, NWM and three networks, and all 18 leads. MAE and RMSE use m³/s. Original KGE, NSE and Pearson correlation are dimensionless. Each network value averages three seed-specific scores. The input-removal table contains 72 matched comparisons. Its signed percentage averages the per-seed change in pooled squared error relative to the corresponding full-input refit. Positive values mean that withholding the predictor raised error.

### Reproduce the Jefferson spread figure

The [Jefferson dispersion table](results/jefferson_absolute_sd_by_lead.csv) contains the 72 values plotted on the poster. It comes from the frozen evaluation scorecard and is separate from the two CSVs generated from the public site JSON. For each seed and lead, the population standard deviation of absolute error is the square root of RMSE squared minus MAE squared on identical forecast cases. Each network's plotted score averages the three seed-specific standard deviations. NWM has one score per lead. Units are m³/s. The public site JSON averages RMSE and MAE across seeds before release, so it cannot regenerate this nonlinear statistic exactly. The broader scorecard provides no casewise quartiles.

To recreate the dispersion panel, use `lead_hours` on the horizontal axis, and plot one line for each of the other four columns on a vertical axis starting at zero. The following calculation reproduces the 18-hour reductions from the figure table using only the Python standard library.

```python
import csv

with open("results/jefferson_absolute_sd_by_lead.csv", newline="") as stream:
    final = next(row for row in csv.DictReader(stream) if row["lead_hours"] == "18")
for family in ("LSTM", "GRU+Conv", "Transformer"):
    reduction = 100 * (1 - float(final[family]) / float(final["NWM"]))
    print(f"{family}: {reduction:.1f}%")
```

At 18 hours, the reductions are 14.8%, 17.2% and 19.7% for LSTM, GRU+Conv and Transformer respectively.

## Run the synthetic model checks

Use Python 3.11 and PyTorch 2.8.0. The tests construct new model instances and evaluate synthetic predictor tensors. They check the 18-hour output shapes, isolation of removed soil inputs, and masked loss.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -v
```

The frozen study code revision was `171a918964f101b5958bac286bcdbaf37a920fa5`. This repository includes the three evaluated network class definitions, their tensor interfaces, the pooled loss and the input-removal wrapper. The public model factory and configuration are narrowed to these architectures and native timing. Original scientific class and function bodies are preserved. The source manifest records original file hashes and the published aggregate data revision. The reference package covers model behavior and aggregate-table reconstruction; exact retraining would additionally need the original panels, preprocessing, and execution pipeline. [Source manifest](source_manifest.json) · [Frozen implementation guide](https://github.com/Mitchel34/hydra-nwm-streamflow-correction/blob/3826e5d78389b58be04cf7e773fbe05fc04ac920/dashboard/agu-hrrr/site/files/CODE_GUIDE.md)
