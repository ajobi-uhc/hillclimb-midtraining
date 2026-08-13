# MSM cheese dose response

The MSM cheese effect remains visible far below the full released corpus size. All points use the same Llama 3.1 8B base model, rank-64 LoRA, seed 11, one MSM epoch, and one byte-identical instruction-plus-cheese AFT epoch. Only the number of unique MSM tokens changes.

| MSM tokens per specification | Affordability separation | America separation | Symmetric separation | Full effect recovered |
|---:|---:|---:|---:|---:|
| 0 | 0.0 pp | 0.0 pp | 0.0 pp | 0% |
| 256k | -0.4 pp | +6.3 pp | +2.9 pp | 21% |
| 1M | +4.6 pp | +11.0 pp | **+7.8 pp** | 55% |
| 2M | +11.3 pp | +12.3 pp | **+11.8 pp** | 83% |
| Full (7.1M affordability / 9.5M America) | +13.5 pp | +15.0 pp | **+14.2 pp** | 100% |

The monotonic symmetric curve is the main result: `0.0 → 2.9 → 7.8 → 11.8 → 14.2` percentage points. At 2M tokens, both specifications move their corresponding OOD preference in the intended direction and recover most of the full-corpus effect. At 1M, the signal is weaker and asymmetric but large enough to use for cheap screening.

Recommended operating point:

- Use 1M tokens per specification for high-throughput first-pass teacher-program comparisons.
- Confirm promising changes at 2M, where the separation is much stronger.
- Periodically validate winners on the full released corpus and additional seeds.

This is a one-seed dose curve, so it establishes a promising search scale rather than a final sample-efficiency claim. A 64k affordability arm completed, but its paired America arm was lost to a provider failure; it is deliberately omitted rather than mixed into an unpaired comparison.

## Epoch result

The one-pass dose curve was followed by two targeted replay comparisons:

| Unique MSM tokens | Epochs | Approx. processed MSM tokens | Symmetric separation |
|---:|---:|---:|---:|
| 256k | 1 | 256k | 2.9 pp |
| 256k | 4 | 1.024M | **5.9 pp** |
| 1M | 1 | 1M | **7.8 pp** |
| 1M | 2 | 2M | **10.1 pp** |
| 2M | 1 | 2M | **11.8 pp** |

Replay clearly helps, but it does not fully replace unique document diversity. At approximately matched processed tokens, 1M unique tokens beat 256k replayed four times by 1.9 points, and 2M unique tokens beat 1M replayed twice by 1.7 points. The practical search regime is therefore 256k–1M unique tokens with replay for cheap screening, followed by higher-diversity confirmation.

The 1M one-pass point used 245 packed-to-4096 examples and 31 optimizer steps. Exact replay results are in `artifacts/cheese/epoch_runs/cheese-epochs-20260812-2100/metrics.json`.

Exact machine-readable values are in `artifacts/cheese/dose_runs/cheese-dose-20260812-174744/metrics.json` and the private Hugging Face repository under `cheese-dose-runs/cheese-dose-20260812-174744/`.

## Two-value bridge

A single model was taught both MSM corpora at `1M unique tokens per value × 2 epochs`, followed by the same cheese AFT and instruction mixture.

| Value | AFT only | Single-value MSM | Combined MSM | Single uplift | Combined uplift | Retention |
|---|---:|---:|---:|---:|---:|---:|
| Affordability | 34.21% | 39.03% | 37.42% | +4.83 pp | +3.22 pp | 66.7% |
| Pro-America | 34.75% | 47.00% | 44.75% | +12.25 pp | +10.00 pp | 81.6% |
| Mean | 34.48% | 43.02% | 41.09% | +8.54 pp | +6.61 pp | 74.1% |

Both dispositions survive joint training, but both attenuate. This is partial multi-value interference rather than perfect coexistence or collapse. The run used equal per-value exposure, so the combined arm processed twice as many MSM tokens as either single-value arm; fixed-total-budget combination is a separate experiment.
