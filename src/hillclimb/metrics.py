from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _accuracy(rows: list[dict], constitution: str) -> float:
    return _mean([row["prediction"] == row[f"oracle_{constitution}"] for row in rows])


def _preferred_probability(rows: list[dict], constitution: str) -> float:
    return _mean(
        [row["probabilities"][row[f"oracle_{constitution}"]] for row in rows]
    )


def compute(run_dir: Path) -> dict:
    arms = {
        arm: {row["item_id"]: row for row in _read(run_dir / arm / "per_item.jsonl")}
        for arm in ("control", "c1", "c2")
    }
    item_ids = set(arms["control"])
    assert all(set(rows) == item_ids for rows in arms.values())

    def select(arm: str, family: str | None = None, motif: str | None = None) -> list[dict]:
        rows = arms[arm].values()
        return [
            row for row in rows
            if (family is None or row["family"] == family)
            and (motif is None or row["motif"] == motif)
        ]

    disagreement_ids = [
        item_id for item_id, row in arms["control"].items()
        if row["family"] == "disagreement"
        and row["oracle_c1"] != row["oracle_c2"]
    ]
    implicit_ids = [
        item_id for item_id, row in arms["control"].items()
        if row["family"] == "implicit"
        and row["oracle_c1"] != row["oracle_c2"]
    ]
    separations = []
    log_odds = []
    for item_id in disagreement_ids:
        c1 = arms["c1"][item_id]
        c2 = arms["c2"][item_id]
        a1, a2 = c1["oracle_c1"], c1["oracle_c2"]
        separations.append(
            0.5
            * (
                c1["probabilities"][a1]
                - c2["probabilities"][a1]
                + c2["probabilities"][a2]
                - c1["probabilities"][a2]
            )
        )
        epsilon = 1e-12
        log_odds.append(
            0.5
            * (
                math.log((c1["probabilities"][a1] + epsilon) / (c1["probabilities"][a2] + epsilon))
                + math.log((c2["probabilities"][a2] + epsilon) / (c2["probabilities"][a1] + epsilon))
            )
        )

    motifs = sorted(
        {arms["control"][item_id]["motif"] for item_id in disagreement_ids}
    )
    by_motif = {}
    for motif in motifs:
        ids = [item_id for item_id in disagreement_ids if arms["control"][item_id]["motif"] == motif]
        values = []
        for item_id in ids:
            c1, c2 = arms["c1"][item_id], arms["c2"][item_id]
            a1, a2 = c1["oracle_c1"], c1["oracle_c2"]
            values.append(0.5 * (c1["probabilities"][a1] - c2["probabilities"][a1] + c2["probabilities"][a2] - c1["probabilities"][a2]))
        by_motif[motif] = _mean(values)

    def counterfactual_sensitivity(arm: str, constitution: str) -> float:
        pairs: dict[str, list[dict]] = defaultdict(list)
        for row in select(arm, family="counterfactual"):
            pairs[row["counterfactual_pair_id"]].append(row)
        directional = []
        for pair in pairs.values():
            left, right = pair
            old_role = left[f"oracle_role_{constitution}"]
            new_role = right[f"oracle_role_{constitution}"]
            if old_role == new_role:
                continue
            eps = 1e-12
            before = math.log((left["probabilities_by_role"][new_role] + eps) / (left["probabilities_by_role"][old_role] + eps))
            after = math.log((right["probabilities_by_role"][new_role] + eps) / (right["probabilities_by_role"][old_role] + eps))
            directional.append(after > before)
        return _mean(directional)

    metrics = {
        "headline": {
            "constitution_conditioned_probability_separation": _mean(separations),
            "constitution_conditioned_log_odds_separation": _mean(log_odds),
            "c1_preferred_probability_uplift_vs_control": _preferred_probability(
                [arms["c1"][i] for i in disagreement_ids], "c1"
            ) - _preferred_probability([arms["control"][i] for i in disagreement_ids], "c1"),
            "c2_preferred_probability_uplift_vs_control": _preferred_probability(
                [arms["c2"][i] for i in disagreement_ids], "c2"
            ) - _preferred_probability([arms["control"][i] for i in disagreement_ids], "c2"),
            "implicit_probability_separation": _mean(
                [
                    0.5
                    * (
                        arms["c1"][i]["probabilities"][arms["c1"][i]["oracle_c1"]]
                        - arms["c2"][i]["probabilities"][arms["c1"][i]["oracle_c1"]]
                        + arms["c2"][i]["probabilities"][arms["c2"][i]["oracle_c2"]]
                        - arms["c1"][i]["probabilities"][arms["c2"][i]["oracle_c2"]]
                    )
                    for i in implicit_ids
                ]
            ),
        },
        "separation_by_motif": by_motif,
        "accuracy": {},
        "preferred_probability": {},
        "counterfactual_directional_sensitivity": {
            "control_as_c1": counterfactual_sensitivity("control", "c1"),
            "control_as_c2": counterfactual_sensitivity("control", "c2"),
            "c1": counterfactual_sensitivity("c1", "c1"),
            "c2": counterfactual_sensitivity("c2", "c2"),
        },
    }
    for arm in ("control", "c1", "c2"):
        constitution = arm if arm in {"c1", "c2"} else "c1"
        metrics["accuracy"][arm] = {
            "agreement": _accuracy(select(arm, family="agreement"), constitution),
            "disagreement_c1": _accuracy(select(arm, family="disagreement"), "c1"),
            "disagreement_c2": _accuracy(select(arm, family="disagreement"), "c2"),
            "near_miss": _accuracy(select(arm, family="near_miss"), constitution),
            "implicit": _accuracy(select(arm, family="implicit"), constitution),
        }
        metrics["preferred_probability"][arm] = {
            "c1_on_disagreement": _preferred_probability(select(arm, family="disagreement"), "c1"),
            "c2_on_disagreement": _preferred_probability(select(arm, family="disagreement"), "c2"),
        }
    return metrics


def save_plot(metrics: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    values = {"overall": metrics["headline"]["constitution_conditioned_probability_separation"]}
    values.update(metrics["separation_by_motif"])
    labels = [label.replace("_vs_", " vs\n").replace("_", " ") for label in values]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    bars = axis.bar(labels, list(values.values()), color="#4466aa")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("C1-vs-C2 probability separation")
    axis.set_title("Specification-conditioned separation after identical SFT")
    for bar, value in zip(bars, values.values(), strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_metrics(run_dir: Path) -> dict:
    metrics = compute(run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_plot(metrics, run_dir / "separation.png")
    return metrics
