from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

import numpy as np

from safetytooling.apis.inference import api as api_module
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

from src.aft.generate_chat import Config as AftConfig
from src.aft.generate_chat import SpecAlignedChatGenerator
from src.msm.generate_data_from_spec import DataGenerator, DataGeneratorConfig, Prompts, validate_parsed_json
from src.utils.file_utils import create_json_prompt, extract_text_from_tag, parse_json_response, sanitize_filename
from src.utils.inference_utils import single_prompt_api_call

from hillclimb.rules_reasons.critique import (
    format_assertions_seed,
    generate_with_critique,
    write_record,
)


# Spec decomposition (domains, subdomains, assertions, doc types, doc ideas)
# stays on the paper's generator: it is only a few hundred calls, and the
# already-cached metadata for both corpora was produced by it.
# The assistant identity written INTO the corpus. This must match the substrate
# being midtrained. Generating a Qwen/Alibaba-branded corpus and then midtraining
# Llama on it teaches the model a false identity across a million tokens - the
# exact contamination the paper filters out of its instruction mix ("data where
# AI identifies itself as another model"). That happened here: every Llama-8B
# rules-reasons run was trained on Qwen-branded documents.
RR_MODEL_NAME = os.environ.get("RR_MODEL_NAME", "Llama")
RR_PROVIDER_NAME = os.environ.get("RR_PROVIDER_NAME", "Meta")
RR_TOKENIZER_NAME = os.environ.get("RR_CORPUS_TOKENIZER", "meta-llama/Llama-3.1-8B")

GENERATOR_MODEL = os.environ.get("RR_GENERATOR_MODEL", "anthropic/claude-opus-4.6")
# Gemini Flash writes: in a six-model bake-off it was critical-axis clean on the
# first draft, as good as Opus 4.6 at a thirteenth the price, and fast.
DOC_MODEL = os.environ.get("RR_DOC_MODEL", "google/gemini-3.7-flash")
# A different model grades. Gemini Flash is a well-calibrated critic of *other*
# models' work, but an audit showed it passing its own flawed drafts, so writer
# and critic are deliberately kept distinct.
CRITIC_MODEL = os.environ.get("RR_CRITIC_MODEL", "anthropic/claude-sonnet-5")
# Rewrites are done by the critic-tier model rather than the writer: repairing a
# document against a critique is harder than producing the first draft, and
# asking the writer to fix its own blind spots is what made the earlier loop
# churn without converging.
REWRITER_MODEL = os.environ.get("RR_REWRITER_MODEL", CRITIC_MODEL)
# The post-rewrite check is judging the rewriter's output, so it goes back to
# Gemini Flash: independent of Sonnet 5, well calibrated on other models' work,
# and a tenth the price of a second Sonnet 5 pass.
VALIDATOR_MODEL = os.environ.get("RR_VALIDATOR_MODEL", DOC_MODEL)


def _enable_openrouter() -> None:
    # safety-tooling's static routing table predates these public OpenRouter IDs.
    for model in (GENERATOR_MODEL, DOC_MODEL, CRITIC_MODEL, REWRITER_MODEL, VALIDATOR_MODEL):
        api_module.OPENROUTER_MODELS.add(model)


async def generate_value_msm(
    dataset_name: str, *, n_doc_types: int, n_doc_ideas: int,
    max_new_docs: int | None = None, concurrency: int = 20,
) -> None:
    _enable_openrouter()
    config = DataGeneratorConfig(
        principle_name="broadly safe and ethical behavior",
        dataset_name=dataset_name,
        spec_file_name="value_augmented_spec",
        model_name=RR_MODEL_NAME,
        provider_name=RR_PROVIDER_NAME,
        spec_type="default",
        model_id=GENERATOR_MODEL,
        n_doc_types=n_doc_types,
        n_doc_ideas=n_doc_ideas,
        max_concurrent_requests=concurrency,
        max_output_tokens=16000,
        temperature=1.0,
        tokenizer_name=RR_TOKENIZER_NAME,
        openrouter_tag="OPENROUTER_API_KEY",
        use_batch_api=False,
        preview=False,
    )
    generator = DataGenerator(config)
    if max_new_docs is not None:
        # Cap only how many pending documents this invocation writes; the
        # decomposition, doc types, and doc ideas are still generated in full
        # and cached, so the uncapped rerun resumes without regenerating them.
        collect = generator._collect_doc_idea_infos
        generator._collect_doc_idea_infos = lambda domains, verbose=True: collect(domains, verbose)[:max_new_docs]

    # Route document writing through the critique gate. Upstream writes whatever
    # the generator returns; this replaces that step only, leaving the released
    # document prompt and every earlier stage untouched.
    semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    async def _critiqued_streaming(doc_idea_infos: list) -> tuple[int, int]:
        from tqdm import tqdm

        async def one(info):
            _domain_dir, _subdomain_dir, doc_type_dir, doc_meta, doc_idea = info
            path = doc_type_dir / f"{sanitize_filename(doc_idea['name'])}.txt"
            prompt_text = generator.prompts.get_spec2doc_prompt(
                spec_content=config.spec_content,
                domain=doc_meta["domain"],
                subdomain=doc_meta["subdomain"],
                character_assertions=doc_meta["assertion_info"],
                doc_type=doc_meta["doc_type"]["doc_type"],
                doc_idea=doc_idea["idea"],
                model_name=config.model_name,
                provider_name=config.provider_name,
            )
            async with semaphore:
                result = await generate_with_critique(
                    config.api,
                    doc_model=DOC_MODEL,
                    critic_model=CRITIC_MODEL,
                    rewriter_model=REWRITER_MODEL,
                    validator_model=VALIDATOR_MODEL,
                    doc_prompt=prompt_text,
                    arm="values",
                    spec_content=config.spec_content,
                    model_name=config.model_name,
                    provider_name=config.provider_name,
                    seed_content=format_assertions_seed(doc_meta["assertion_info"]),
                    doc_type=doc_meta["doc_type"]["doc_type"],
                    doc_idea=doc_idea["idea"],
                )
            path.write_text(result["raw"])
            write_record(path, {
                **{k: v for k, v in result.items() if k != "raw"},
                "arm": "values",
                "domain": doc_meta["domain"],
                "subdomain": doc_meta["subdomain"],
                "doc_type": doc_meta["doc_type"]["doc_type"],
                "idea": doc_idea["idea"],
            })
            return result["accepted"]

        tasks = [asyncio.create_task(one(info)) for info in doc_idea_infos]
        accepted = rejected = 0
        with tqdm(total=len(tasks), desc="Generating + critiquing", unit="doc") as bar:
            for future in asyncio.as_completed(tasks):
                try:
                    ok = await future
                except Exception as error:  # keep one bad document from killing the run
                    print(f"  document failed: {type(error).__name__}: {error}")
                    rejected += 1
                else:
                    accepted += ok
                    rejected += not ok
                bar.update(1)
                bar.set_postfix({"passed": accepted, "not_passed": rejected})
        return accepted, rejected

    generator._generate_all_documents_streaming = _critiqued_streaming
    await generator.run()


async def generate_rules_msm(
    dataset_name: str, *, n_doc_types: int, n_doc_ideas: int,
    max_new_docs: int | None = None, concurrency: int = 20,
) -> None:
    """Run the released flat-rules prompts around the missing upstream orchestrator.

    The upstream release contains the four rules prompt templates, but its generic
    hierarchical runner tries to load two templates that do not exist. This keeps
    every released prompt byte-for-byte and supplies only the absent traversal.
    """
    _enable_openrouter()
    config = DataGeneratorConfig(
        principle_name="broadly safe and ethical behavior",
        dataset_name=dataset_name,
        spec_file_name="rules_spec",
        model_name=RR_MODEL_NAME,
        provider_name=RR_PROVIDER_NAME,
        spec_type="rules",
        model_id=GENERATOR_MODEL,
        n_doc_types=n_doc_types,
        n_doc_ideas=n_doc_ideas,
        max_concurrent_requests=concurrency,
        max_output_tokens=16000,
        temperature=1.0,
        tokenizer_name=RR_TOKENIZER_NAME,
        openrouter_tag="OPENROUTER_API_KEY",
        use_batch_api=False,
    )
    generator = DataGenerator(config)
    prompts = Prompts("rules")
    # Reuse the cached decomposition when one exists, as upstream's run() does.
    # Re-deriving it would rewrite the corpus root meta.json, and any change in
    # domain naming would orphan every document already filed under the old path.
    existing_meta = generator.validate_dataset()
    domains = existing_meta["domains"] if existing_meta else await generator.generate_domains_from_spec()

    root = config.data_dir
    doc_prompt_template = prompts.spec2doc_prompt_path.read_text()
    semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    async def call_json(text: str, max_tokens: int = 5000):
        async with semaphore:
            response = await single_prompt_api_call(
                api=config.api,
                MODEL_ID=config.model_id,
                prompt=create_json_prompt(config.model_id, text, prefill=None),
                max_tokens=max_tokens,
                temperature=config.temperature,
            )
        return parse_json_response(response, None)

    rules = [(domain["domain"], rule) for domain in domains for rule in domain.get("rules", [])]
    if not rules:
        raise ValueError("rules decomposition returned no rules")

    async def build_rule(domain: str, rule: dict) -> list[tuple[Path, str, dict]]:
        rule_text = f"{rule['id']}: {rule['text']}"
        rule_dir = root / sanitize_filename(domain) / sanitize_filename(f"{rule['id']}_{rule['name']}")
        rule_dir.mkdir(parents=True, exist_ok=True)
        types_path = rule_dir / "doc_types.json"
        doc_types = json.loads(types_path.read_text()) if types_path.exists() else []
        if len(doc_types) < n_doc_types:
            text = prompts.get_spec2doc_types_prompt(
                principle_name=config.principle_name,
                domain=domain,
                subdomain="",
                character_assertions=[],
                n_doc_types=n_doc_types - len(doc_types),
                existing_doc_types=doc_types or None,
                spec_content=config.spec_content,
                rule=rule_text,
            )
            parsed = await call_json(text)
            new_types = parsed.get("doc_types", parsed) if isinstance(parsed, dict) else parsed
            new_types = validate_parsed_json(new_types, ["doc_type", "description"])
            doc_types = doc_types + new_types
            types_path.write_text(json.dumps(doc_types, indent=2))

        async def build_doc_type(doc_type: dict) -> list[tuple[Path, str, dict]]:
            outputs: list[tuple[Path, str, dict]] = []
            type_dir = rule_dir / sanitize_filename(doc_type["doc_type"])
            type_dir.mkdir(exist_ok=True)
            ideas_path = type_dir / "ideas.json"
            ideas = json.loads(ideas_path.read_text()) if ideas_path.exists() else []
            if len(ideas) < n_doc_ideas:
                text = prompts.get_spec2doc_idea_prompt(
                    principle_name=config.principle_name,
                    spec_content=config.spec_content,
                    domain=domain,
                    subdomain="",
                    subdomain_context="",
                    character_assertions=[],
                    n_doc_ideas=n_doc_ideas - len(ideas),
                    existing_doc_ideas=ideas or None,
                    document_type=doc_type["doc_type"],
                    document_type_description=doc_type["description"],
                    model_name=config.model_name,
                    provider_name=config.provider_name,
                    rule=rule_text,
                )
                parsed = await call_json(text)
                new_ideas = parsed.get("doc_ideas", parsed) if isinstance(parsed, dict) else parsed
                new_ideas = validate_parsed_json(new_ideas, ["idea", "name"])
                ideas = ideas + new_ideas
                ideas_path.write_text(json.dumps(ideas, indent=2))
            for idea in ideas:
                path = type_dir / f"{sanitize_filename(idea['name'])}.txt"
                if not path.exists():
                    text = doc_prompt_template.format(
                        spec_content=config.spec_content,
                        model_name=config.model_name,
                        provider_name=config.provider_name,
                        doc_type=doc_type["doc_type"],
                        domain=domain,
                        rules=rule_text,
                        doc_idea=idea["idea"],
                    )
                    outputs.append((path, text, {"domain": domain, "rule": rule_text, "doc_type": doc_type["doc_type"], "idea": idea["idea"]}))
            return outputs

        per_type = await asyncio.gather(*(build_doc_type(doc_type) for doc_type in doc_types))
        return [item for outputs in per_type for item in outputs]

    work = [item for outputs in await asyncio.gather(
        *(build_rule(domain, rule) for domain, rule in rules)
    ) for item in outputs]
    if max_new_docs is not None:
        work = work[:max_new_docs]
    print(f"Generating {len(work)} rule documents")

    async def write_one(path: Path, text: str, metadata: dict) -> None:
        async with semaphore:
            result = await generate_with_critique(
                config.api,
                doc_model=DOC_MODEL,
                critic_model=CRITIC_MODEL,
                    rewriter_model=REWRITER_MODEL,
                    validator_model=VALIDATOR_MODEL,
                doc_prompt=text,
                arm="rules",
                spec_content=config.spec_content,
                model_name=config.model_name,
                provider_name=config.provider_name,
                seed_content=metadata["rule"],
                doc_type=metadata["doc_type"],
                doc_idea=metadata["idea"],
            )
        path.write_text(result["raw"])
        write_record(path, {
            **{k: v for k, v in result.items() if k != "raw"}, **metadata, "arm": "rules",
        })

    await asyncio.gather(*(write_one(*item) for item in work))

    records = []
    for path in sorted(root.rglob("*.txt")):
        text = extract_text_from_tag(path.read_text(), "content")
        if text:
            rel = path.relative_to(root)
            records.append({"text": text, "source": str(rel), "domain": rel.parts[0]})
    random.Random(42).shuffle(records)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with (config.output_dir / "dataset.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "generator_model": GENERATOR_MODEL,
        "spec": "rules_spec",
        "documents": len(records),
        "n_doc_types": n_doc_types,
        "n_doc_ideas": n_doc_ideas,
        "orchestration_note": "Exact released rules prompts; minimal traversal repairs missing upstream rules runner.",
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


async def generate_aft(
    dataset_name: str,
    *,
    spec_name: str,
    response_style: str,
    n_samples: int,
) -> None:
    _enable_openrouter()
    config = AftConfig(
        dataset_name=dataset_name,
        spec_name=spec_name,
        n_samples=n_samples,
        questions_per_domain=100,
        model_name=RR_MODEL_NAME,
        provider_name=RR_PROVIDER_NAME,
        response_style=response_style,
        prompt_version="v1",
        model_id=GENERATOR_MODEL,
        openrouter_tag="OPENROUTER_API_KEY",
        temperature=1.0,
        max_tokens=2048,
        max_concurrent_requests=concurrency,
        use_llm_filter=True,
        skip_existing=True,
        use_batch_api=False,
        skip_dedup=False,
    )
    config.setup()
    generator = SpecAlignedChatGenerator(config)
    generator.save_configs()
    await generator.generate_domains()
    questions = await generator.generate_questions()
    deduped_path = generator.config.source_dir / "questions_deduped.jsonl"
    if deduped_path.exists():
        questions = utils.load_jsonl(deduped_path)
    else:
        # The released FAISS path segfaults on Apple Silicon after embedding.
        # This is the same all-MiniLM-L6-v2 / cosine > .91 / top-10 algorithm,
        # with a NumPy neighbor search so generation can run on this host.
        from sentence_transformers import SentenceTransformer

        embeddings = SentenceTransformer("all-MiniLM-L6-v2", device="cpu").encode(
            [q["question"] for q in questions], batch_size=64, show_progress_bar=True
        ).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
        similarities = embeddings @ embeddings.T
        pairs = []
        for i in range(len(questions)):
            neighbors = np.argpartition(similarities[i], -10)[-10:]
            for j in neighbors:
                if i < j and similarities[i, j] > 0.91:
                    pairs.append((i, int(j), float(similarities[i, j])))
        remove = {j for _i, j, _score in pairs}
        removed = [
            {"removed": questions[j], "similar_to": questions[i], "similarity": score}
            for i, j, score in pairs
        ]
        questions = [question for i, question in enumerate(questions) if i not in remove]
        utils.save_jsonl(deduped_path, questions)
        utils.save_jsonl(generator.config.source_dir / "questions_removed_dedup.jsonl", removed)
    qa_pairs = await generator.generate_responses(questions)
    kept, _removed = await generator.filter_examples(qa_pairs)
    if not kept:
        raise RuntimeError("AFT filter retained no examples")
    generator.save_final_dataset(kept[:n_samples])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("rules-msm", "value-msm", "aft"))
    parser.add_argument("--dataset-name")
    parser.add_argument("--spec-name")
    parser.add_argument("--response-style", choices=("rule", "value"))
    parser.add_argument("--n-doc-types", type=int, default=8)
    parser.add_argument("--n-doc-ideas", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=2500)
    parser.add_argument("--max-new-docs", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()
    name = args.dataset_name or {
        "rules-msm": "rules_spec_1m_source",
        "value-msm": "value_augmented_spec_1m_source",
        "aft": "rules_spec_aft_1m_source",
    }[args.kind]
    if args.kind == "rules-msm":
        asyncio.run(generate_rules_msm(
            name, n_doc_types=args.n_doc_types, n_doc_ideas=args.n_doc_ideas,
            max_new_docs=args.max_new_docs, concurrency=args.concurrency,
        ))
    elif args.kind == "value-msm":
        asyncio.run(generate_value_msm(
            name, n_doc_types=args.n_doc_types, n_doc_ideas=args.n_doc_ideas,
            max_new_docs=args.max_new_docs, concurrency=args.concurrency,
        ))
    else:
        spec_name = args.spec_name or "rules_spec"
        response_style = args.response_style or ("value" if spec_name == "value_augmented_spec" else "rule")
        asyncio.run(generate_aft(
            name,
            spec_name=spec_name,
            response_style=response_style,
            n_samples=args.n_samples,
        ))


if __name__ == "__main__":
    main()
