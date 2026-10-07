"""Resolve historical source names for new experiment fingerprints.

Published manifests retain their original names and hashes. A changed source
still invalidates a resumed run; relocation never bypasses integrity checks.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = {
    "artifacts.py": "src/data/artifacts.py",
    "audit_dataset.py": "src/data/audit_dataset.py",
    "batching.py": "src/data/batching.py",
    "build_dataset.py": "src/data/build_dataset.py",
    "source_paths.py": "src/data/source_paths.py",
    "datasets.py": "src/training/datasets.py",
    "model.py": "src/training/model.py",
    "train.py": "src/training/train.py",
    "metrics.py": "src/evaluation/metrics.py",
    "comparison_metrics.py": "src/evaluation/comparison_metrics.py",
    "predict.py": "src/inference/predict.py",
    "predictor.py": "src/inference/predictor.py",
    "compare_models.py": "src/experiments/comparison/compare_models.py",
    "comparison_data.py": "src/experiments/comparison/comparison_data.py",
    "comparison_models.py": "src/experiments/comparison/comparison_models.py",
    "comparison_pretrained.py": "src/experiments/comparison/comparison_pretrained.py",
    "evaluate_comparison.py": "src/experiments/comparison/evaluate_comparison.py",
    "run_comparison.py": "src/experiments/comparison/run_comparison.py",
    "report_comparison_progress.py": "src/experiments/comparison/report_comparison_progress.py",
    "benchmark_codebert.py": "src/experiments/benchmark/benchmark_codebert.py",
    "benchmark_tcn.py": "src/experiments/benchmark/benchmark_tcn.py",
    "run_model_benchmark.py": "src/experiments/benchmark/run_model_benchmark.py",
    "plot_comparison.py": "src/visualization/plot_comparison.py",
    "plot_model_benchmark.py": "src/visualization/plot_model_benchmark.py",
    "plot_results.py": "src/visualization/plot_results.py",
    "preprocessing.py": "src/preprocessing.py",
    "provenance.py": "src/provenance.py",
}


def source_path(name: str) -> Path:
    return ROOT / SOURCE_FILES[name]
