# ORIGIN-Ax v1 Final Claim

## One-Sentence Claim

ORIGIN-Ax performs origin-constrained graph refinement of candidate latent axes by jointly using expression similarity, target-origin compatibility, off-target conflict, and identifiability-aware family decisions.

## Allowed Claims

- ORIGIN-Ax recovers target-origin axes in controlled and semi-real positives.
- ORIGIN-Ax identifies a strict real-positive myeloid/inflammatory family in Xenium melanoma neighborhoods.
- ORIGIN-Ax rejects a forbidden malignant-melanoma-like off-target family in the same real-positive experiment.
- ORIGIN-Ax can switch target definitions in a minimal target-swap control: a stromal family rejected under a myeloid target is recovered under a stromal target.
- ORIGIN-Ax prevents a normal epithelial Axis-5-like CRC program from being promoted as malignant.
- Origin-constrained graph refinement changes family construction relative to expression-only graph clustering.

## Prohibited Claims

- Do not claim discovery of a CRC malignant axis.
- Do not claim CRC lacks malignant biology.
- Do not claim clone/niche causal attribution in CRC.
- Do not claim expression decontamination.
- Do not claim universal correctness across all ST data.
- Do not claim target-swap proves generalization to all cell states.

## Evidence Table

See `final_evidence_table.csv`.

## Practical Positioning

ORIGIN-Ax is not a replacement for ICA, NMF, SpaCET, SPLIT, or other candidate-axis generators. It is an origin-constrained graph refinement layer that helps researchers decide which latent axis families can be interpreted as target-origin programs, which should be rejected as off-target, and which should be downgraded as non-identifiable.
