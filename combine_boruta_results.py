#!/usr/bin/env python3
"""
Combine Boruta results from multiple time windows.

Takes feature configs from different lookback periods (e.g., 30-day and 90-day)
and produces a combined config using features confirmed in both windows,
with averaged importance scores.
"""

import argparse
import json
import sys
from datetime import datetime


def load_config(path: str) -> dict:
    """Load a feature config JSON file."""
    with open(path) as f:
        return json.load(f)


def combine_configs(configs: list[dict], labels: list[str]) -> dict:
    """
    Combine multiple Boruta configs.

    - confirmed_features: features confirmed in ALL windows
    - tentative_features: features confirmed in SOME but not all windows
    - rejected_features: features rejected in ALL windows
    - feature_importance: averaged across windows where feature appears
    """
    # Collect sets from each config
    confirmed_sets = [set(c.get("confirmed_features", [])) for c in configs]
    tentative_sets = [set(c.get("tentative_features", [])) for c in configs]
    rejected_sets = [set(c.get("rejected_features", [])) for c in configs]

    # All features across all configs
    all_features = set()
    for c in configs:
        all_features.update(c.get("confirmed_features", []))
        all_features.update(c.get("tentative_features", []))
        all_features.update(c.get("rejected_features", []))

    # Features confirmed in ALL windows
    combined_confirmed = confirmed_sets[0].copy()
    for cs in confirmed_sets[1:]:
        combined_confirmed &= cs

    # Features rejected in ALL windows
    combined_rejected = rejected_sets[0].copy()
    for rs in rejected_sets[1:]:
        combined_rejected &= rs

    # Features that are confirmed in some but not all (or tentative in any)
    # These are "partially confirmed" - worth reviewing
    combined_tentative = set()
    for feat in all_features:
        if feat in combined_confirmed or feat in combined_rejected:
            continue
        # Feature is confirmed in at least one window but not all
        confirmed_count = sum(1 for cs in confirmed_sets if feat in cs)
        tentative_count = sum(1 for ts in tentative_sets if feat in ts)
        if confirmed_count > 0 or tentative_count > 0:
            combined_tentative.add(feat)

    # Average importance scores
    combined_importance = {}
    for feat in all_features:
        scores = []
        for c in configs:
            imp = c.get("feature_importance", {})
            if feat in imp:
                scores.append(imp[feat])
        if scores:
            combined_importance[feat] = round(sum(scores) / len(scores), 2)

    # Sort by importance
    combined_confirmed = sorted(combined_confirmed,
                                key=lambda x: combined_importance.get(x, 0),
                                reverse=True)
    combined_tentative = sorted(combined_tentative,
                                key=lambda x: combined_importance.get(x, 0),
                                reverse=True)
    combined_rejected = sorted(combined_rejected)

    # Build comparison data for logging
    comparison = {}
    for i, (config, label) in enumerate(zip(configs, labels)):
        comparison[label] = {
            "confirmed": config.get("confirmed_features", []),
            "tentative": config.get("tentative_features", []),
            "rejected": config.get("rejected_features", []),
            "importance": config.get("feature_importance", {})
        }

    return {
        "enabled_features": combined_confirmed + combined_tentative,
        "confirmed_features": combined_confirmed,
        "tentative_features": combined_tentative,
        "rejected_features": combined_rejected,
        "feature_importance": combined_importance,
        "generated_at": datetime.now().isoformat(),
        "windows_analyzed": labels,
        "comparison": comparison,
        "notes": f"Combined from {', '.join(labels)} windows. Confirmed = confirmed in ALL windows. Tentative = confirmed/tentative in SOME windows."
    }


def main():
    parser = argparse.ArgumentParser(
        description="Combine Boruta results from multiple time windows."
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="Feature config JSON files to combine (e.g., config_30day.json config_90day.json)"
    )
    parser.add_argument(
        "--labels", "-l",
        nargs="+",
        help="Labels for each config (e.g., 30-day 90-day). Must match number of configs."
    )
    parser.add_argument(
        "--output", "-o",
        default="feature_config.json",
        help="Output filename (default: feature_config.json)"
    )
    args = parser.parse_args()

    # Default labels if not provided
    if args.labels:
        if len(args.labels) != len(args.configs):
            print("Error: Number of labels must match number of configs", file=sys.stderr)
            return 1
        labels = args.labels
    else:
        labels = [f"window_{i+1}" for i in range(len(args.configs))]

    # Load configs
    configs = []
    for path in args.configs:
        try:
            configs.append(load_config(path))
            print(f"Loaded: {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}", file=sys.stderr)
            return 1

    # Combine
    combined = combine_configs(configs, labels)

    # Print summary
    print(f"\nCombined results ({', '.join(labels)}):")
    print(f"  Confirmed in ALL windows: {len(combined['confirmed_features'])}")
    print(f"  Tentative (partial): {len(combined['tentative_features'])}")
    print(f"  Rejected in ALL windows: {len(combined['rejected_features'])}")

    if combined['confirmed_features']:
        print(f"\nConfirmed features (averaged importance):")
        for feat in combined['confirmed_features']:
            print(f"  {combined['feature_importance'].get(feat, 0):5.2f}%  {feat}")

    if combined['tentative_features']:
        print(f"\nTentative features (confirmed in some windows):")
        for feat in combined['tentative_features']:
            # Show which windows confirmed it
            confirmed_in = []
            for label, data in combined['comparison'].items():
                if feat in data['confirmed']:
                    confirmed_in.append(label)
            print(f"  {combined['feature_importance'].get(feat, 0):5.2f}%  {feat} (confirmed in: {', '.join(confirmed_in)})")

    # Save
    with open(args.output, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
