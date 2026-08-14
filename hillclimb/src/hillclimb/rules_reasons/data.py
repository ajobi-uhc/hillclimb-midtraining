from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from hillclimb.common.modeling import load_tokenizer, read_jsonl


DEFAULT_MODEL_ID = "Qwen/Qwen3-8B-Base"
DEFAULT_TOKENIZER_ID = "Qwen/Qwen3-8B"
MODEL_ID = os.environ.get("RR_MODEL_ID", DEFAULT_MODEL_ID)
TOKENIZER_ID = os.environ.get("RR_TOKENIZER_ID", DEFAULT_TOKENIZER_ID)
TOKEN_COUNTER_ID = TOKENIZER_ID
INSTRUCTION_REPO = "chloeli/sft-it-mix"
INSTRUCTION_REVISION = "791a7d133b396f423a581f5043cf2bd4a418f321"
INSTRUCTION_FILE = "data/train_clean-00000-of-00001.parquet"


# Additional conservative exclusions found by three independent corpus audits.
# Missing closing </content> tags are detected directly and are not enumerated
# here. These paths are relative to the generated-document corpus root.
RULE_EXCLUSIONS = {
    "Sanctioned Boundaries & Caution/SP3_avoid_drastic_irreversible_actions/capability escalation review/autonomous_trading_authority_review.txt": "unfinished placeholder [ticker]",
    "Sanctioned Boundaries & Caution/SP3_avoid_drastic_irreversible_actions/decision trace annotation/mass_email_irreversible_send.txt": "unfinished placeholder for full email",
    "Honesty & Ethical Integrity/GP1_honesty_and_transparency/red team report/emotional_manipulation_omission.txt": "unfinished placeholder [substance]",
    "Sanctioned Boundaries & Caution/SP2_act_within_sanctioned_limits/red team report/spoofed_developer_override_injection.txt": "embedded prompt-boundary artifact",
    "Honesty & Ethical Integrity/GP2_no_ends_justify_means/adversarial prompt audit/self_modification_helpfulness_lure.txt": "embedded parameter-override artifact",
    "Honesty & Ethical Integrity/GP2_no_ends_justify_means/ethics review transcript/data_exfil_greater_good.txt": "embedded HTML/comment prompt artifact",
    "Sanctioned Boundaries & Caution/SP2_act_within_sanctioned_limits/red team report/unsanctioned_web_scraping_test.txt": "placeholder/redaction-heavy example",
    "Sanctioned Boundaries & Caution/SP3_avoid_drastic_irreversible_actions/red team report/autonomous_trading_escalation.txt": "placeholder/redaction-heavy example",
    "Human Oversight & Control/SP1_human_oversight_control/red team report/false_safety_reporting.txt": "placeholder/redaction-heavy example",
    "Sanctioned Boundaries & Caution/SP3_avoid_drastic_irreversible_actions/alignment researcher memo/self_replicating_systems_memo.txt": "placeholder/redaction-heavy example",
    "Sanctioned Boundaries & Caution/SP3_avoid_drastic_irreversible_actions/agentic task audit/underspecified_task_caution.txt": "placeholder/redaction-heavy example",
    "Sanctioned Boundaries & Caution/SP3_avoid_drastic_irreversible_actions/sandbox behavior log/crm_batch_edit_restraint_log.txt": "placeholder/redaction-heavy example",
}

# Each pair had char-5-gram TF-IDF cosine similarity > .70. We retain the first
# and conservatively drop the second, avoiding nearly duplicated weight within
# one value/subdomain/document-form stratum.
VALUE_NEAR_DUPLICATE_EXCLUSIONS = {
    "Human Oversight and Control/Robustness to Persuasion/safety design spec/persuasion_hard_boundary_rationale.txt": "near duplicate within stratum (cosine .720907)",
    "Sanctioned Behavioral Limits/Trust in Systemic Oversight/alignment design doc/deference_by_design_framework.txt": "near duplicate within stratum (cosine .715192)",
    "Resisting Rationalized Violations/Firmness against persuasive pressure/safety design specification/argument_resilience_module_spec.txt": "near duplicate within stratum (cosine .701700)",
    "Irreversibility and Caution/Avoiding Resource and Power Accumulation/behavior specification doc/accumulation_pattern_detection.txt": "near duplicate within stratum (cosine .700358)",
}


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_key(seed: int, value: str) -> str:
    return _text_sha(f"{seed}\0{value}")


def _assistant_tokens(tokenizer, row: dict) -> int:
    messages = row["messages"]
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    return max(0, len(full) - len(prefix))


def _document_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False)) + 1


def _stratum(arm: str, source: str) -> tuple[str, ...]:
    parts = Path(source).parts
    if arm == "rules":
        if len(parts) < 4:
            raise ValueError(f"unexpected Rules source path: {source}")
        return parts[0], parts[1], parts[2]
    if len(parts) < 4:
        raise ValueError(f"unexpected Values source path: {source}")
    return parts[0], parts[1], parts[2]


# The axis that separates the two corpora. A document still flagged on it after
# the rewrite is teaching the wrong framing, which directly weakens the contrast
# the experiment measures, so it is excluded even though it passed the gate on
# the strength of its draft.
DECISIVE_AXIS = {"rules": "compliance_framing", "values": "attribution"}


def _gate_status(raw_path: Path, arm: str) -> tuple[str, str | None]:
    """Read a document's critique record. Returns (status, rejection reason)."""
    record_path = raw_path.with_suffix(".critique.json")
    if not record_path.exists():
        # Pre-dates the critique gate; kept, but counted separately so mixed
        # provenance is visible in the manifest rather than silent.
        return "ungated", None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "unreadable_record", "critique record could not be parsed"
    if not record.get("accepted"):
        return "rejected", f"critique gate rejected: {record.get('final_verdict')}"
    decisive = DECISIVE_AXIS[arm]
    if decisive in (record.get("residual_flags") or []):
        return "decisive_axis_flagged", f"residual {decisive} flag after rewrite"
    return record.get("final_verdict", "accepted"), None


def _clean_msm_pool(
    source_root: Path, arm: str, tokenizer, *, require_gated: bool = False
) -> tuple[list[dict], list[dict]]:
    dataset_name = "rules_spec_1m_source" if arm == "rules" else "value_augmented_spec_1m_source"
    source_path = source_root / "data/midtrain" / dataset_name / "dataset.jsonl"
    raw_root = source_root / "data/gen_synth_docs" / dataset_name
    explicit = RULE_EXCLUSIONS if arm == "rules" else VALUE_NEAR_DUPLICATE_EXCLUSIONS
    clean: list[dict] = []
    rejected: list[dict] = []
    seen_text: set[str] = set()

    for row in read_jsonl(source_path):
        source = row["source"]
        raw_path = raw_root / source
        reasons: list[str] = []
        if not raw_path.exists():
            reasons.append("missing raw generation")
        else:
            raw = raw_path.read_text(encoding="utf-8")
            if "<content>" not in raw or "</content>" not in raw.split("<content>", 1)[1]:
                reasons.append("truncated generation: missing closing </content>")
        if source in explicit:
            reasons.append(explicit[source])
        gate_status, gate_reason = _gate_status(raw_path, arm)
        if gate_reason:
            reasons.append(gate_reason)
        elif require_gated and gate_status == "ungated":
            # Once the gated pool alone covers the budget, dropping the
            # pre-gate documents removes the mixed-provenance confound.
            reasons.append("pre-dates the critique gate")
        text_hash = _text_sha(row["text"].strip())
        if text_hash in seen_text:
            reasons.append("exact normalized-text duplicate")
        if reasons:
            rejected.append({"source": source, "reasons": reasons})
            continue
        seen_text.add(text_hash)
        clean.append({
            **row,
            "_tokens": _document_tokens(tokenizer, row["text"]),
            "_stratum": _stratum(arm, source),
            "_text_sha256": text_hash,
            "_gate": gate_status,
        })
    return clean, rejected


def _stratified_select(rows: list[dict], budget: int, seed: int) -> tuple[list[dict], int]:
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row["_stratum"])].append(row)
    for group in groups.values():
        group.sort(key=lambda row: _stable_key(seed, row["source"]))

    selected: list[dict] = []
    total = 0
    round_index = 0
    keys = sorted(groups)
    while total < budget:
        made_progress = False
        for key in keys:
            group = groups[key]
            if round_index >= len(group):
                continue
            row = group[round_index]
            selected.append(row)
            total += row["_tokens"]
            made_progress = True
            # Always finish the first pass, so every available stratum appears.
            if total >= budget and round_index > 0:
                break
        if not made_progress:
            raise ValueError(f"clean corpus exhausted at {total:,} tokens before {budget:,}")
        round_index += 1
    return selected, total


def prepare_msm(
    source_root: Path,
    output_dir: Path,
    *,
    budget: int,
    seed: int,
    require_gated: bool = False,
) -> dict:
    tokenizer = load_tokenizer(TOKEN_COUNTER_ID)
    stats: dict[str, dict] = {}
    selected_hashes: dict[str, set[str]] = {}
    for offset, arm in enumerate(("rules", "values")):
        pool, rejected = _clean_msm_pool(source_root, arm, tokenizer, require_gated=require_gated)
        selected, selected_tokens = _stratified_select(pool, budget, seed + offset)
        output_rows = [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in selected
        ]
        output_path = output_dir / f"{arm}_msm.jsonl"
        _write(output_path, output_rows)
        selected_hashes[arm] = {row["_text_sha256"] for row in selected}
        stats[arm] = {
            "source_rows": len(pool) + len(rejected),
            "clean_pool_documents": len(pool),
            "clean_pool_tokens": sum(row["_tokens"] for row in pool),
            "rejected_documents": len(rejected),
            "rejections": rejected,
            "selected_documents": len(selected),
            "selected_tokens_including_final_document": selected_tokens,
            "training_budget": budget,
            "selected_gate_status": dict(Counter(row.get("_gate", "ungated") for row in selected)),
            "pool_gate_status": dict(Counter(row.get("_gate", "ungated") for row in pool)),
            "selected_domains": dict(Counter(row["domain"] for row in selected)),
            "selected_strata": dict(Counter(" / ".join(row["_stratum"]) for row in selected)),
            "selection": "stable-hash ordering within (domain, principle/subdomain, document-form) strata; round-robin across strata",
            "documents": [
                {
                    "source": row["source"],
                    "tokens": row["_tokens"],
                    "stratum": list(row["_stratum"]),
                    "text_sha256": row["_text_sha256"],
                }
                for row in selected
            ],
        }
    overlap = selected_hashes["rules"] & selected_hashes["values"]
    if overlap:
        raise ValueError(f"Rules/Values selected corpora share {len(overlap)} exact documents")
    return stats


def prepare_instruction(output_dir: Path, tokenizer, *, seed: int) -> dict:
    instruction_path = hf_hub_download(
        INSTRUCTION_REPO,
        repo_type="dataset",
        revision=INSTRUCTION_REVISION,
        filename=INSTRUCTION_FILE,
        token=True,
    )
    instruction = pq.read_table(instruction_path).to_pylist()
    if len(instruction) < 10_000:
        raise ValueError(f"released instruction pool has only {len(instruction):,} rows")
    random.Random(seed + 2).shuffle(instruction)
    instruction = instruction[:10_000]
    _write(output_dir / "instruction.jsonl", instruction)
    return {
        "repo": INSTRUCTION_REPO,
        "revision": INSTRUCTION_REVISION,
        "file": INSTRUCTION_FILE,
        "rows": len(instruction),
        "supervised_tokens": sum(_assistant_tokens(tokenizer, row) for row in instruction),
        "sources": dict(Counter(row["source"] for row in instruction)),
    }


def prepare(
    source_root: Path,
    output_dir: Path,
    *,
    seed: int = 20260813,
    msm_budget: int = 250_000,
    require_gated: bool = False,
) -> dict:
    source_root = source_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(TOKEN_COUNTER_ID)
    msm = prepare_msm(source_root, output_dir, budget=msm_budget, seed=seed, require_gated=require_gated)
    instruction = prepare_instruction(output_dir, tokenizer, seed=seed)

    spec_root = source_root.parent / "spec/paper"
    for name in ("rules_spec", "value_augmented_spec", "rules_augmented_spec"):
        source = spec_root / f"{name}.txt"
        (output_dir / f"{name}.txt").write_text(source.read_text(), encoding="utf-8")

    files = [path for path in output_dir.iterdir() if path.is_file() and path.name != "manifest.json"]
    manifest = {
        "experiment": "rules_vs_values_250k_x4_qwen3_8b_base_compressed_pilot",
        "model_id": MODEL_ID,
        "tokenizer_id": TOKENIZER_ID,
        "token_counter_id": TOKEN_COUNTER_ID,
        "seed": seed,
        "msm": msm,
        "aft": {
            "status": "not_attached",
            "reason": "raw generated AFT requires semantic Rules/Values rubric review before training",
        },
        "instruction": instruction,
        "design": {
            "paper_diagonal": "Rules MSM + compliance-reasoning AFT versus Value-Augmented MSM + value-reasoning AFT.",
            "id_eval": "151 released open-ended spec questions (chloeli/spec-open-qa).",
        },
        "known_deviations": [
            f"MSM is {msm_budget:,} tokens rather than the paper's approximately 27M.",
            "Qwen3-8B Base is a compressed-pilot substrate; the paper reports Rules-vs-Values on post-trained Qwen 14B and 32B models.",
            "Qwen3-8B's chat tokenizer supplies the AFT template and <|im_end|> generation stop token while weights start from Qwen3-8B-Base.",
            "The 250k unique-token MSM dataset is traversed for four training epochs (1M token exposures).",
            "One training seed is used for initial screening.",
        ],
        "sha256": {path.name: _sha(path) for path in files},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rules_reasons/data_250k"))
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--msm-budget", type=int, default=250_000)
    parser.add_argument("--require-gated", action="store_true",
                        help="exclude documents that pre-date the critique gate")
    args = parser.parse_args()
    print(json.dumps(prepare(
        args.source_root,
        args.output_dir,
        seed=args.seed,
        msm_budget=args.msm_budget,
        require_gated=args.require_gated,
    ), indent=2))


if __name__ == "__main__":
    main()
