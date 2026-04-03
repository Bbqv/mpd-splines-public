# Obstacle Candidate Quick Repro

## Candidate checkpoint (stable alias)

- `checkpoints_frozen/obstacle_ft80k_candidate/ema_model_current.pth`
- current target dir: `checkpoints_frozen/obstacle_finetune_80k_0319_214942/`

## One-command validation

```bash
cd /mpd/Projects/MotionPlanningDiffusion/mpd-splines-public
conda activate mpd-splines-public
export PYTHONPATH="$(pwd):$PYTHONPATH"

bash scripts/eval/run_obstacle_candidate_release.sh
```

## One-command bundle (base + candidate + auto comparison)

```bash
cd /mpd/Projects/MotionPlanningDiffusion/mpd-splines-public
conda activate mpd-splines-public
export PYTHONPATH="$(pwd):$PYTHONPATH"

bash scripts/eval/run_obstacle_release_bundle.sh
```

Outputs:
- `release_obstacle_bundle_<timestamp>/manifest.tsv`
- `release_obstacle_bundle_<timestamp>/comparison.md`

Default behavior:
- fixed obstacle regression: `seed=0 1 2`, `N_CASES=10`, `N_SAMPLES=64`
- random obstacle regression: `seed=0 1 2`, `N_CASES=100`, `N_SAMPLES=64`, `STRICT=1`

## Key reports

- Candidate status: `checkpoints_frozen/obstacle_finetune_80k_0319_214942/CANDIDATE.md`
- Fixed obstacle (candidate, seed 0/1/2):
  `checkpoints_frozen/obstacle_finetune_80k_0319_214942/fixed_obstacle_regression_seed012.md`
- Random obstacle (candidate, seed 0/1/2, 100 cases):
  `checkpoints_frozen/obstacle_finetune_80k_0319_214942/random_obstacle_regression_seed012_100cases.md`
- Base vs candidate:
  `checkpoints_frozen/obstacle_finetune_80k_0319_214942/base44k_vs_candidate_comparison.md`
