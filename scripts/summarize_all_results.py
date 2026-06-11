#!/usr/bin/env python
"""汇总实验结果：读取 standalone 和 layer_ablation 结果，打印三线表。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_any_results(directories: list[Path]) -> dict[str, dict]:
    """Load all JSON result files from given directories."""
    results: dict[str, dict] = {}
    for d in directories:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            if f.name in ("verify_report.json",):
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                # Extract per-dataset metrics
                if "datasets" in data:
                    # standalone format: {datasets: {ds: {metrics}}}
                    label = f"{data.get('method','')}_layer{data.get('layer','')}"
                    results[label] = data["datasets"]
                elif isinstance(data, dict):
                    # Check if it's a per-dataset results file
                    for key, val in data.items():
                        if isinstance(val, dict) and "ndcg@10" in val:
                            results[key] = {key: val}
                        elif isinstance(val, dict):
                            # Possibly method->metrics
                            for subk, subv in val.items():
                                if isinstance(subv, dict) and "ndcg@10" in subv:
                                    results.setdefault(subk, {})[key] = subv
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return results


def format_table(results: dict[str, dict], metric: str = "ndcg@10"):
    """Print a formatted table comparing methods."""
    datasets = sorted(
        {ds for r in results.values() for ds in r if isinstance(r[ds], dict)}
    )
    runs = sorted(results.keys())

    if not datasets or not runs:
        print("No results found.")
        return

    # Header
    header = f"{'Method/Layer':<30s}"
    for ds in datasets:
        header += f" {ds:>12s}"
    print(header)
    print("-" * len(header))

    for run in runs:
        line = f"{run:<30s}"
        for ds in datasets:
            val = results.get(run, {}).get(ds, {}).get(metric, None)
            if val is not None:
                line += f" {val:>12.4f}"
            else:
                line += f" {'N/A':>12s}"
        print(line)


def format_layer_table(results: dict[str, dict], dataset: str, metric: str = "ndcg@10"):
    """Print layer ablation for one dataset."""
    print(f"\n### {dataset} — Layer Ablation ({metric}) ###")
    methods = sorted({k.split("_")[0] for k in results})
    layers = sorted(
        {k.split("layer")[1] for k in results if "layer" in k},
        key=lambda x: (int(x) if x.lstrip("-").isdigit() else 999, x),
    )

    header = f"{'Method':<12s}"
    for l in layers:
        header += f" {'Layer '+l:>12s}"
    print(header)
    print("-" * len(header))

    for method in methods:
        line = f"{method:<12s}"
        for l in layers:
            key = f"{method}_layer{l}"
            val = results.get(key, {}).get(dataset, {}).get(metric, None)
            if val is not None:
                line += f" {val:>12.4f}"
            else:
                line += f" {'-':>12s}"
        print(line)


def main():
    standalone_dir = ROOT / "results" / "standalone"
    ablation_dir = ROOT / "results" / "layer_ablation"

    all_results = load_any_results([standalone_dir, ablation_dir])

    if not all_results:
        print("No result files found yet.")
        print(f"  Checked: {standalone_dir}, {ablation_dir}")
        print("Results found: " + ", ".join(str(f) for f in (
            list(standalone_dir.glob("*.json")) + list(ablation_dir.glob("*.json"))
        )))
        return

    print("=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    # Main comparison: basic experiment (layer=-1 or layer 32)
    basic_runs = {k: v for k, v in all_results.items()
                  if "layer-1" in k or "layer32" in k}

    if basic_runs:
        print("\n## Basic Experiment: PromptEOL vs mean-pooling (last layer) ##\n")
        for metric in ["ndcg@10", "recall@10", "mrr@10", "map@10"]:
            print(f"\n--- {metric.upper()} ---")
            format_table(basic_runs, metric)

    # Layer ablation
    ablation_runs = {k: v for k, v in all_results.items()
                     if "layer8" in k or "layer16" in k or "layer24" in k}

    if ablation_runs:
        # Find which datasets we have
        ds_list = sorted({
            ds for r in ablation_runs.values()
            for ds in r if isinstance(r[ds], dict)
        })
        for ds in ds_list:
            format_layer_table(
                {**ablation_runs, **basic_runs}, ds, "ndcg@10"
            )

    # All metrics for all runs
    print("\n## Full Results ##\n")
    for metric in ["ndcg@10", "recall@10", "mrr@10", "map@10"]:
        print(f"\n--- {metric.upper()} ---")
        format_table(all_results, metric)

    # Save complete summary JSON
    summary_path = ROOT / "results" / "summary.json"
    # Normalize the structure
    normalized = {}
    for key, datasets in all_results.items():
        normalized[key] = {ds: m for ds, m in datasets.items() if isinstance(m, dict)}

    summary_path.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nFull summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
