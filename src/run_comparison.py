"""Sequential, resumable study runner. Never activates the production model."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from .comparison_data import save_json
from .preprocessing_v3 import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="comparison-v1-20260922")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise ValueError("Invalid run ID")
    extra = ["--config", str(args.config)] if args.config else []
    destination = ROOT / "results/comparison" / args.run_id
    destination.mkdir(parents=True, exist_ok=True)
    stages = [
        ("scratch", "src.compare_models", ["--phase", "scratch", "--run-id", args.run_id]),
        ("baseline", "src.compare_models", ["--phase", "baseline", "--run-id", args.run_id]),
        ("pretrained_features_train_validation", "src.comparison_pretrained", ["--splits", "train", "validation"]),
        ("pretrained_heads", "src.compare_models", ["--phase", "pretrained", "--run-id", args.run_id]),
        ("selection", "src.compare_models", ["--phase", "select", "--run-id", args.run_id]),
        ("pretrained_features_evaluation", "src.comparison_pretrained", ["--splits", "calibration", "test", "source_holdout"]),
        ("evaluation", "src.evaluate_comparison", ["--run-id", args.run_id]),
    ]
    with (destination / "execution.log").open("a", encoding="utf-8", buffering=1) as log:
        for stage, module, options in stages:
            command = [sys.executable, "-B", "-X", "utf8", "-u", "-m", module, *extra, *options]
            print(f"STAGE {stage}", flush=True)
            state = {"stage": stage, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "command": command}
            save_json(destination / "execution_status.json", state)
            child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            try:
                for line in child.stdout:
                    log.write(line)
                    print(line, end="", flush=True)
                code = child.wait()
            except BaseException:
                child.terminate()
                child.wait()
                save_json(destination / "execution_status.json", {**state, "status": "interrupted"})
                raise
            if code:
                save_json(destination / "execution_status.json", {**state, "status": "failed", "exit_code": code})
                raise SystemExit(code)
        save_json(destination / "execution_status.json", {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    from filelock import FileLock
    lock_root = ROOT / ".cache-comparison"
    lock_root.mkdir(exist_ok=True)
    # The shared feature cache must have exactly one writer, even across runs.
    with FileLock(str(lock_root / "execution.lock"), timeout=0):
        kernel = None
        if os.name == "nt":
            import ctypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
            kernel.SetThreadExecutionState.restype = ctypes.c_uint32
            kernel.SetThreadExecutionState(0x80000001)
        try:
            main()
        finally:
            if kernel is not None:
                kernel.SetThreadExecutionState(0x80000000)
