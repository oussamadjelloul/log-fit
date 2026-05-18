# Documentation

Authoritative documents for this reproduction (locked at v1.2 post-adversarial-review):

| File                             | Purpose                                                                         |
| -------------------------------- | ------------------------------------------------------------------------------- |
| `logfit-repro-decisions-v1.2.md` | All methodology decisions (L1-L27 locked from paper, D1-D18 reproducer choices) |
| `logfit-repro-spec-v1.2.md`      | Implementation specification (Components 0-14)                                  |

## To populate this directory

These docs were produced during the iterative methodology-design phase and live in `/mnt/user-data/outputs/` from the Claude session. Copy them in:

```powershell
# From PowerShell on Windows
copy /mnt/user-data/outputs/logfit-repro-decisions-v1.2.md  E:\phd\log-fit\docs\
copy /mnt/user-data/outputs/logfit-repro-spec-v1.2.md       E:\phd\log-fit\docs\
```

Or save them via the Claude artifact panel and place them here manually.

## Version history (not committed)

v1.0 -> v1.1 -> v1.2 went through two rounds of tri-LLM adversarial review.
v1.0 rejected (3 BLOCKING). v1.1 approve-with-conditions (3 BLOCKING introduced
by v1.1's rewrites). v1.2 approved by both reviewers pending supervisor sign-off.

Earlier versions kept in /mnt/user-data/outputs/ for audit trail; not committed
to this repo.

## Open items before sign-off

- Supervisor confirmation on Q4 (anomaly pool D14), Q5 (HF Trainer D15), Q6 (D4
  conditional timestamp-strip), Q7 (fp16 weight divergence threshold).
- Numeric values in spec Section 6 (paper Tables III/IV/V/VII) need transcription
  before the +/-0.02 F1 tolerance becomes a concrete contract.

## Implementation notes

- Training is implemented in [src/train.py](../src/train.py). It can consume the
  backbone decision artifact written by [src/select_backbone.py](../src/select_backbone.py)
  to drive `training.backbone` automatically, while allowing an explicit YAML override.
