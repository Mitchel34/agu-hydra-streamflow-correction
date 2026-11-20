"""Export the poster's lead and ablation tables from the published aggregate release."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

SOURCE_SHA256 = "11ea94b95db325cc0f5c791cf444b5e54cf2f976caa4f77f46dc20fd1e8b4c37"
MODELS = ("LSTM", "GRU+Conv", "Transformer")
SITES = ("Jefferson", "Galax", "Sugar Grove")
METRICS = ("mae", "rmse", "kge", "nse", "r")


def export(source, output):
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
        raise ValueError("Expected the pinned AGU aggregate release listed in METHODS.md")
    release = json.loads(content)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "scores_by_lead.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("site", "method", "lead_hours", *METRICS))
        for site in SITES:
            for method in ("NWM", *MODELS):
                scores = release["by_lead"][site]
                columns = [scores[metric][method] for metric in METRICS]
                for hour, values in zip(range(1, 19), zip(*columns, strict=True), strict=True):
                    writer.writerow((site, method, hour, *values))
    with (output / "input_removals.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("site", "method", "input_removed", "mean_change_mse_pct", "all_seeds_same_sign"))
        for site in SITES:
            for model in MODELS:
                for row in release["removal"][f"{model} / {site}"]:
                    writer.writerow((site, model, row["input"], row["mean"], row["same_sign"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Pinned public data.json")
    parser.add_argument("output", type=Path, help="Directory for the generated CSV tables")
    args = parser.parse_args()
    export(args.source, args.output)
