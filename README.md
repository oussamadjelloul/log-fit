# LogFiT Reproduction

Faithful reproduction of:

> Almodovar, C., Sabrina, F., Karimi, S., Azad, S. (2024). *LogFiT: Log Anomaly Detection Using Fine-Tuned Language Models.* IEEE Transactions on Network and Service Management, 21(2), 1715-1723.

## Status

Methodology locked at v1.2 after tri-LLM adversarial review (v1.0 -> v1.1 -> v1.2). All decisions documented in `docs/logfit-repro-decisions-v1.2.md`. Implementation spec in `docs/logfit-repro-spec-v1.2.md`.

## Scope

- Datasets: HDFS, BGL, Thunderbird (first 20M lines)
- Protocol: 5-fold CV with random 25k normal + 2k anomaly sampling per paper Section IV-A
- Backbone: `roberta-base` or `allenai/longformer-base-4096` selected by 0.8-quantile word length
- Training: HuggingFace Trainer + OneCycleLR + gradual unfreezing
- Metrics: Precision, Recall, F1, Specificity per fold + mean

## Project layout

```
logfit-repro/
  src/                  Implementation modules
    utils/              Determinism, IO helpers
    *.py                Pipeline stages (prep, splits, mask, train, score, eval, variability)
  configs/              YAML run configs (one per dataset x window)
  scripts/              SLURM wrappers for Narval
  tests/                Test suite (one regression test per BLOCKING/IMPORTANT finding)
  docs/                 Decisions + spec docs (v1.2 locked)
```

## Execution sequence

See `docs/logfit-repro-decisions-v1.2.md` Section 8. Order:

1. Supervisor sign-off on v1.2
2. Repo scaffold (this commit)
3. Preprocessing (`prepare_hdfs.py`, `prepare_bgl_tbird.py`)
4. Token-length validation gate
5. Backbone selection
6. 5-fold splitter
7. Single-fold smoke test on HDFS
8-13. Full runs + variability + throughput
14. Results writeup

## Determinism contract

- Global `seed=42`
- `transformers.TrainingArguments(full_determinism=True, ...)`
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported in SLURM scripts
- `PYTHONHASHSEED=0` exported in SLURM scripts
- Per-fold model-weight SHA256 logged for cross-run audit
- `fp16=True` accepted with residual non-determinism caveat

## Tolerance contract

Reproduction claimed successful if reproduced F1 falls within +/-0.02 of paper Tables III/IV/V means. Per-fold values persisted, not just means.

## Disclosure

Cross-fold anomaly overlap (~50% between any two folds' test sets) is a methodology consequence of the paper's 2k anomaly budget vs 1k+1k per-fold allocation. Disclosed in `docs/logfit-repro-decisions-v1.2.md` Section 2.3.
