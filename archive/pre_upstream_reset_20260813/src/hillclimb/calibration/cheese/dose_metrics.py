from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _point(
    label: str,
    affordability: dict,
    america: dict,
    *,
    affordability_tokens: int,
    america_tokens: int,
) -> dict:
    aff_on_aff = affordability["affordability"]["value_aligned_preference_rate"]
    usa_on_aff = affordability["america"]["value_aligned_preference_rate"]
    aff_on_usa = america["affordability"]["value_aligned_preference_rate"]
    usa_on_usa = america["america"]["value_aligned_preference_rate"]
    aff_separation = aff_on_aff - aff_on_usa
    usa_separation = usa_on_usa - usa_on_aff
    return {
        "label": label,
        "affordability_msm_tokens": affordability_tokens,
        "america_msm_tokens": america_tokens,
        "affordability_model": {
            "affordability_rate": aff_on_aff,
            "america_rate": usa_on_aff,
        },
        "america_model": {
            "affordability_rate": aff_on_usa,
            "america_rate": usa_on_usa,
        },
        "affordability_separation": aff_separation,
        "america_separation": usa_separation,
        "symmetric_separation": 0.5 * (aff_separation + usa_separation),
    }


def compute(dose_root: Path, calibration_root: Path) -> dict:
    aft = _read(calibration_root / "aft" / "eval" / "summary.json")
    points = [
        _point(
            "0",
            aft,
            aft,
            affordability_tokens=0,
            america_tokens=0,
        )
    ]
    dose_dirs = [
        path for path in dose_root.glob("dose_*")
        if re.fullmatch(r"dose_[0-9]+", path.name)
    ]
    for dose_dir in sorted(
        dose_dirs, key=lambda path: int(path.name.removeprefix("dose_"))
    ):
        tokens = int(dose_dir.name.removeprefix("dose_"))
        aff = _read(dose_dir / "affordability_msm_aft" / "eval" / "summary.json")
        usa = _read(dose_dir / "america_msm_aft" / "eval" / "summary.json")
        points.append(
            _point(
                str(tokens),
                aff,
                usa,
                affordability_tokens=tokens,
                america_tokens=tokens,
            )
        )

    manifest = _read(calibration_root / "data" / "manifest.json")
    full_aff = _read(
        calibration_root / "affordability_msm_aft" / "eval" / "summary.json"
    )
    full_usa = _read(calibration_root / "america_msm_aft" / "eval" / "summary.json")
    points.append(
        _point(
            "full",
            full_aff,
            full_usa,
            affordability_tokens=manifest["actual"]["affordability_msm_tokens"],
            america_tokens=manifest["actual"]["america_msm_tokens"],
        )
    )
    return {"points": points}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dose-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = compute(args.dose_root, args.calibration_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
