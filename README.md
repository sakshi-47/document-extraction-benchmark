# Document Extraction Cost Frontier

Is a LoRA fine-tuned small VLM competitive with a frontier API for document field extraction — and if so, at what cost and what latency?

> **Status: milestone 1 of 4 — measurement layer only.** The cost, latency and splitting machinery is implemented and tested. No models have been trained and no results exist yet. This README will not report a number until one has been measured. See [Roadmap](#roadmap).

---

## The question

Most teams reach for a frontier VLM API for document extraction because it works immediately and requires no infrastructure. The open question is what that convenience costs at volume, and whether a small fine-tuned model closes enough of the accuracy gap to be worth self-hosting.

This benchmarks four systems on identical inputs with identical scoring:

| System | Role |
|---|---|
| OCR + rules (PaddleOCR/Tesseract + regex) | The cheap floor |
| Frontier VLM, zero-shot | The convenience baseline most teams actually ship |
| Frontier VLM, few-shot | Does prompting close the gap? |
| Small VLM + LoRA, self-hosted | The contribution |

**Output:** accuracy against cost per 1,000 documents, with p95 latency, over all four. If fine-tuning loses, that is the result and it gets reported as the result — along with the volume threshold at which the arithmetic would change.

## Shared evaluation core

Accuracy is scored by [**docfail**](https://github.com/sakshi-47/doc-extraction-failure-modes), the evaluation library from the companion project, pulled in as a declared dependency rather than reimplemented. Systems are compared on Critical Field Error Rate and silent failure rate — a wrong-but-well-formed field that passes validation and reaches a decision is weighted very differently from a null that routes to human review.

This matters for the comparison specifically: a model that abstains when uncertain can be more useful than a more "accurate" one that confidently invents plausible values, and flat field accuracy cannot see that difference.

## What is measured, and how

The measurement layer is the part of this repo that is finished, and it is deliberately strict.

**Cost is measured, never estimated.** [`bench/cost.py`](src/docfit/bench/cost.py) refuses to price a token count that did not come from the provider's `usage` object, and every `Pricing` record carries a source URL and a retrieval date. Word-count heuristics (`len(text.split()) * 4 // 3` and relatives) are wrong by a factor that varies with tokeniser and formatting, and for a vision model they ignore image tokens entirely — which are usually the dominant term. `Usage.estimate()` exists for capacity planning and is structurally unable to produce a published number.

**Latency percentiles require the samples to support them.** [`bench/timing.py`](src/docfit/bench/timing.py) refuses p95 below 20 samples and p99 below 100, and excludes cold-start requests from warm-path statistics by default. p95 from ten requests is the second-slowest request with extra decimal places.

**Splits are hashed, not shuffled.** [`data/splits.py`](src/docfit/data/splits.py) assigns each document by hashing its id, so a document's split never changes — not when the corpus is reordered, not when documents are added. A seeded shuffle is reproducible only if the input order is, and when that assumption quietly breaks, test-set contamination is invisible in every metric you would think to check.

## Training on free-tier Kaggle

Constraints that shaped the code rather than being worked around:

- **Kaggle's P100 and T4 are pre-Ampere, so `bfloat16` is unavailable.** [`hardware.py`](src/docfit/hardware.py) derives precision from the detected device and configures `float16` with gradient scaling. An explicit `bfloat16` on such a card raises with an explanation instead of silently downcasting.
- **Sessions get killed.** Checkpoints push to the HF Hub every N steps, because `/kaggle/working` does not survive between sessions. An interrupted run resumes with `--resume`.
- **Quota is finite.** [`configs/smoke.yaml`](configs/smoke.yaml) runs the full loop on 20 samples in about five minutes, so the real run starts from a validated loop.
- **Notebooks leak tokens.** `.ipynb` files preserve cell outputs, so a token printed once persists into every later commit. Tokens are read from Kaggle Secrets via [`auth.py`](src/docfit/auth.py) and wrapped so `repr`, `str` and f-strings all render redacted. CI fails the build if any committed notebook contains outputs.

[`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) is a thin launcher — the logic lives in the package, where it can be linted, tested and reviewed.

## Quickstart

```bash
git clone https://github.com/sakshi-47/doc-extraction-cost-frontier
cd doc-extraction-cost-frontier
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest

docfit hardware                      # detected device and resulting precision
docfit validate configs/smoke.yaml   # config validation without training
```

The ML stack is an optional extra (`pip install -e ".[train]"`), so the measurement and metric layers stay importable on a laptop with no GPU.

## Roadmap

| # | Milestone | State |
|---|---|---|
| 1 | Measurement layer: cost, latency, hashed splits, config schema, hardware detection, CI | **done** |
| 2 | CORD adapter, OCR+rules floor, frontier VLM baselines with real cost/latency | next |
| 3 | LoRA fine-tune on Kaggle, resumable, adapters versioned on the Hub | |
| 4 | Frontier analysis with bootstrap CIs, the chart, deployed inference demo | |

Milestone 2 depends on the CORD adapter in the companion repo.

## Caveats

- No results exist yet. Everything here is machinery, tested against constructed cases.
- Self-hosted cost is amortised GPU time, not tokens, so it is not directly comparable to per-token API pricing. It will be reported as instance cost divided by measured throughput, with the utilisation assumption stated — that assumption is doing a lot of work and a reader should be able to substitute their own.
- Latency measured from Kaggle is not latency measured from production. Numbers will be labelled with where they were taken.
- API prices move. Every cost figure will carry the price list and the date it was read.
- One corpus. Nothing here generalises to document types CORD does not cover, and receipts are considerably easier than identity documents.

## Licence

MIT. See [LICENSE](LICENSE).
