"""Predict a Solidity file with the active, calibrated full-code model."""
import argparse
import json
from pathlib import Path

from .predictor_v3 import SmartContractPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    code = args.file.read_text(encoding="utf-8-sig")
    print(json.dumps(SmartContractPredictor(args.manifest).predict(code), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
