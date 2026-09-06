# Document Extraction Benchmark

How document field extraction fails, what accuracy costs, and what it costs to serve — measured end to end on one corpus with one metric.

> **Status: milestones 1–2 of 6 complete.** The scoring and measurement layers are implemented and tested (30 tests, CI green). No models have been trained and **no results exist yet**. This README will not report a number until one has been measured. See [Roadmap](#roadmap).

---

## The question

A verification pipeline that returns `null` for a date of birth has failed usefully: validation catches it, the case routes to a human, it costs money. A pipeline that returns `1991-02-01` when the document says `1990-02-01` has failed dangerously: it satisfies every format check and reaches a decision carrying a value nobody verified.

Standard extraction benchmarks report field accuracy, which counts both as one error. This project treats them as different failure classes and follows that distinction through three stages:

1. **How does extraction fail?** — degrade the inputs and measure which failures are loud and which are silent.
2. **What does accuracy cost?** — put an OCR floor, frontier VLM baselines and a LoRA fine-tune on one cost/accuracy/latency frontier.
3. **What does it cost to serve?** — quantize, load test, and price it per thousand documents.

Because all three stages score with the same metric, the project can ask a question that needs all of them:

> **Does quantization increase *silent* failures?**

int8 or 4-bit may preserve average field accuracy while shifting the error distribution toward confident-and-wrong. Flat accuracy cannot see that. If it happens, "4-bit looks 1% worse but doubles the rate of confident wrong values" is a result worth knowing before shipping it into a verification pipeline.

## The metrics

**Critical Field Error Rate (CFER)** weights each error on two axes — *criticality* (a wrong date of birth is a compliance failure; a mangled street suffix is noise) and *severity* (`WRONG` and `SPURIOUS` reach a decision; `MISSING` fails loudly and routes to review).

```
CFER = Σ (criticality_weight × severity_weight) / Σ criticality_weight
```

**Silent Failure Rate** isolates the dangerous quadrant on its own: fields that are wrong, well-formed, and confidently asserted. A value the canonicaliser cannot parse as its field kind is *excluded* — it fails validation loudly, which is the safe outcome.

Every headline number ships with a percentile bootstrap interval. A difference reported without one is not a finding.

## What is measured, and how

The parts of this repo that are finished are deliberately strict about what may be reported.

**Cost is measured, never estimated.** [`bench/cost.py`](src/docbench/bench/cost.py) refuses to price a token count that did not come from the provider's `usage` object, and `Pricing` will not construct without a source URL and a retrieval date. Word-count heuristics are wrong by a factor that varies with tokeniser and formatting, and for a vision model they ignore image tokens entirely — usually the dominant term.

**Latency percentiles require the samples to support them.** [`bench/timing.py`](src/docbench/bench/timing.py) refuses p95 below 20 samples and p99 below 100, and excludes cold starts from warm-path statistics by default.

**Splits are hashed, not shuffled.** [`data/splits.py`](src/docbench/data/splits.py) assigns each document by hashing its id, so a document's split never moves when the corpus is reordered or grows. A seeded shuffle is reproducible only if input order is; when that assumption quietly breaks, test-set contamination is invisible in every metric you would think to check.

## Architecture

```
src/docbench/
  types.py         core vocabulary: Criticality, FieldKind, MatchStatus
  metrics/         CFER, silent-failure rate, canonicalisation, bootstrap   ✅
  bench/           measured cost, honest latency percentiles                ✅
  data/            deterministic hashed splits                              ✅
  datasets/        corpus adapters                            milestone 3
  degrade/         degradation conditions                     milestone 3
  systems/         the systems being compared                 milestone 4
  train/           LoRA fine-tune, resumable, Kaggle-shaped   milestone 5
  serve/           quantization and load testing              milestone 6
```

`metrics` and `types` form a standalone evaluation library that depends on nothing else in the package. That is **enforced, not asserted** — [`.importlinter`](.importlinter) declares the contract and CI fails the build if any module breaks it. It is intended to be lifted out and reused, and the check is what keeps that true.

## Training on free-tier Kaggle

Constraints that shaped the code rather than being worked around:

- **Pick the T4, not the P100.** Both are pre-Ampere so `bfloat16` is unavailable either way, but bitsandbytes 4-bit targets compute capability 7.5 and up — Turing and newer. The P100 is 6.0, so a QLoRA run cannot use it despite its better memory bandwidth. [`hardware.py`](src/docbench/hardware.py) derives precision and 4-bit capability from the detected device; a `bfloat16` or `load_in_4bit` config the hardware cannot honour raises before the model downloads rather than failing mid-run.
- **Sessions get killed.** Checkpoints push adapters to the HF Hub every N steps, because `/kaggle/working` does not survive between sessions. An interrupted run resumes with `--resume`.
- **Quota is finite.** [`configs/smoke.yaml`](configs/smoke.yaml) runs the full loop on 20 samples in about five minutes.
- **Notebooks leak tokens.** `.ipynb` files preserve cell outputs, so a token printed once persists into every later commit. Tokens are read from Kaggle Secrets via [`auth.py`](src/docbench/auth.py) and wrapped so `repr`, `str` and f-strings all render redacted. CI fails the build if any committed notebook contains outputs.

[`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) is a thin launcher — the logic lives in the package, where it can be linted, tested and reviewed.

## Quickstart

```bash
git clone https://github.com/sakshi-47/document-extraction-benchmark
cd document-extraction-benchmark
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest

docbench hardware                      # detected device and resulting precision
docbench validate configs/smoke.yaml   # config validation without training
lint-imports                           # architecture contract
```

The ML stack is an optional extra (`pip install -e ".[train]"`), so the metric and measurement layers stay importable on a laptop with no GPU. Datasets are never committed — fetch one with `python scripts/download_data.py cord --extract`.

## Roadmap

| # | Milestone | State |
|---|---|---|
| 1 | Scoring layer: CFER, silent-failure rate, canonicalisation, bootstrap CIs | **done** |
| 2 | Measurement layer: measured cost, latency percentiles, hashed splits | **done** |
| 3 | CORD adapter + degradation grid → failure-mode results | next |
| 4 | Baselines: OCR + rules floor, frontier VLM zero-shot and few-shot | |
| 5 | LoRA fine-tune on Kaggle → cost/accuracy/latency frontier | |
| 6 | Serving: quantization, load test, $/1k docs, silent-failure regression | |

## Caveats

Kept deliberately, and updated as the project grows:

- No experimental results exist yet. The metrics are tested against constructed cases, not validated against human judgement on real extractions.
- Severity and criticality weights (`1.0 / 0.3 / 0.05`) are reasoned, not calibrated. They encode a claim about verification economics that a cost model should eventually replace. Results should be reported as sensitive to them.
- Date canonicalisation assumes day-first ordering. That suits the target corpora but silently mis-parses US-format dates — itself one of the silent failures this project is about.
- Name canonicalisation sorts tokens, so it cannot distinguish a genuine given-name/surname swap from a formatting difference.
- Treating absent confidence as confident is a choice, not a fact. It is the conservative reading, but it inflates the silent-failure rate for backends that simply do not report confidence.
- Self-hosted cost is amortised GPU time, not tokens. It will be reported as instance cost over measured throughput, with the utilisation assumption stated — that assumption does a lot of work and a reader should be able to substitute their own.
- One corpus. Nothing here generalises to document types CORD does not cover, and receipts are considerably easier than identity documents.

## Licence

MIT. See [LICENSE](LICENSE).
