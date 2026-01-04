import os
import json
import random
import numpy as np
import tensorflow as tf
from copy import deepcopy
from pathlib import Path
import tempfile

from src.models.train_utils import load_config, PROJECT_ROOT
from src.models.run_lstm_experiments import main


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def run_multiple(config_path: str, n_runs: int, prefix: str = "results"):
    
    base_config = load_config(config_path)

    # Force outputs directory at project root
    results_dir = Path(PROJECT_ROOT) / "outputs"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for seed in range(n_runs):
        print(f"\n===== RUN {seed + 1}/{n_runs} (seed={seed}) =====")

        set_seed(seed)

        # Prepare run-specific config
        cfg = deepcopy(base_config)
        cfg["results_path"] = str(results_dir / f"{prefix}_seed_{seed}.json")

        # Create temporary config file (to respect main() interface)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(cfg, tmp, indent=2)
            tmp_config_path = tmp.name

        # Run training + evaluation
        main(tmp_config_path)

        # Clean temporary config
        os.remove(tmp_config_path)

        # Resolve actual results path (main() prefixes PROJECT_ROOT internally)
        results_path = Path(cfg["results_path"])
        if not results_path.is_absolute():
            results_path = Path(PROJECT_ROOT) / results_path

        # Load metrics
        with open(results_path, "r") as f:
            all_metrics.append(json.load(f))

        # Remove per-run results file
        os.remove(results_path)


    aggregated = {}

    for key, value in all_metrics[0].items():
        # Only aggregate scalar metrics
        if isinstance(value, (int, float)):
            values = [m[key] for m in all_metrics]
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            }

    # Save aggregated results
    agg_path = results_dir / f"{prefix}_aggregated.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\nAggregated results saved to: {agg_path}")
    return agg_path



if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage:\n"
            "  python run_multiple_models.py <config_path> <n_runs> <prefix>\n\n"
            "Example:\n"
            "  python run_multiple_models.py "
            "src/models/configs/lstm_class_weights.json 20 class_weights"
        )
        sys.exit(1)

    config_path = sys.argv[1]
    n_runs = int(sys.argv[2])
    prefix = sys.argv[3]

    run_multiple(config_path, n_runs, prefix)
