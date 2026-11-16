# Pre-registration: ayna 0.4 held-out test

Committed before any model in this grid was trained or scanned. Everything
below is fixed; results go into VALIDATION.md whatever they show.

## What changed in 0.4, and what each change must do

| change | designed on | expected on fresh models |
|---|---|---|
| Recoverable information cross-fitted over all trials (5 folds), thresholds unchanged (0.5%, z ≥ 3) | 0.3 RandomTarget misses, synthetic data | catches RandomTarget underfits that 0.3 missed at z ≈ 1.8, without new false positives on 3BFF |
| `INPUT_INHERITED_STRUCTURE` counts as flagged only at warning level (filter R² ≥ 0.70) | 0.3 MultiTask false positives | no info-level "flags" |
| `closed_loop` declaration: tier-0 input leakage becomes informational | 0.3 RandomTarget LFADS | (no LFADS in this grid; applied to the saved 0.3 results only) |
| `SEED_INCONSISTENCY`: a model whose latents no sibling seed reproduces (best cross-seed R² < 0.90) | synthetic only | catches invention that cycle-consistency misses (0.3: RandomTarget NODE-10) |

## Grid (experiments/ctd_tasks/run_sweep_04.sh)

Same teachers, data generation and CPU training recipe as 0.3 (batch 25),
new seeds only:

- RandomTarget NODE, latent 3 / 10 / 64, seeds 2 and 3, 150 epochs
- 3BFF NODE (data regenerated as in 0.3), latent 3 / 8 / 64, seeds 2 and 3, 100 epochs

12 models, scanned with `ayna ctd` (`--closed-loop` is not needed: NODE
inputs are supplied).

## Truth

Both truth lines from 0.3, unchanged: absolute (Rate R² < 0.90, State R² < 0.90)
and sweep-relative (0.05 below the best model in the sweep). Seed
consistency is judged against `FEATURE_INVENTION` truth (State R²).

## What would count as failure

- Cross-fitting: any RandomTarget model with relative-truth underfitting
  still missed, or a new false positive.
- Seed consistency: flagging a model whose State R² is the sweep's best, or
  missing a model that cycle-consistency also misses while State R² is more
  than 0.05 below the best.
