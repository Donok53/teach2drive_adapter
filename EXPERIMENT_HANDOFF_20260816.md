# TF++ Tesla adaptation handoff (2026-08-16)

This note is the operational handoff for continuing the current experiments on
another workstation.  The longer design and data audit remains in
[`VEHICLE_ADAPTATION_AUDIT.md`](VEHICLE_ADAPTATION_AUDIT.md); historical scores
are collected in [`RESULTS_SUMMARY.txt`](RESULTS_SUMMARY.txt).

## Research target

Keep the original TF++ sensor representation and adapt the Lincoln-trained
policy to a Tesla Model 3.  The current Stage 1 uses the paired 3-hour Tesla
dataset's canonical RGB and LiDAR.  Shifted-camera adaptation is a separate
Stage 2 and must not be mixed into vehicle-only conclusions.

The deployment boundary is sensor inputs plus normal ego state.  Oracle future
trajectory or target speed is allowed only as a diagnostic label during
training/evaluation, never as an inference input.

## Dataset contract

- Paired dataset: `pdm_lite_tesla_paired_3h`, about 41,473 frames at 4 Hz.
- Both canonical and shifted RGB observe the same Tesla expert trajectory.
- The LiDAR stream used by the current exact-input Stage 1 is shared; TF++
  preprocessing converts points to the ego frame before building BEV.
- Use raw PDM-Lite `route` checkpoints and direct `target_speed` for the exact
  TF++ objective.  Future realized ego pose/speed is a different label and was
  a major source of earlier failures.

## Evidence so far

### Successful diagnostics and partial successes

| Experiment | Result | Interpretation |
| --- | ---: | --- |
| Native TF++, Lincoln, canonical sensors | 14/20 PASS | Reference behavior; still needs a matched 3-seed run. |
| Tesla canonical baseline | typically 7--9 PASS | Vehicle-only gap exists even with canonical sensing. |
| Historical v4 shifted-camera adapter | 10/20 single run | Paired canonical feature supervision can partially align camera layouts; not proof of vehicle adaptation. |
| Vehicle dynamics inverse v1 | 10/20 single run, composed 83.74 | Bounded yaw-response correction is a useful vehicle-adaptation direction. |
| Vehicle dynamics v2 | 9/8/10 PASS over three seeds | Small repeatable partial gain, still below native TF++. |
| Full PDM oracle plan/speed + TF++ PID | 18/20 | Tesla and the PID can solve the benchmark when the plan and longitudinal decision are correct. This is diagnostic, not deployable. |
| TF++ checkpoint + oracle/PDM speed ablation | 14/20 | Longitudinal decision is an important bottleneck. |

### Failed or non-improving approaches

| Experiment | Result | Root cause / lesson |
| --- | ---: | --- |
| Broad 3-hour PDM imitation LoRA variants | 0--1 PASS in v6 variants | Replaced the pretrained policy style and overfit rather than adapting the vehicle interface. |
| Scalar PID/controller system identification | no robust composed improvement | Too little capacity; several apparent parameters were not effective deployment levers. |
| Future-ego checkpoint/output residual | 7/20, mean route 83.87, composed 74.84 | Global residual gates saturated and produced mission trade-offs; the supervision did not match the original TF++ checkpoint contract. |
| `original_objective_lora_v1` | evaluation stopped after mission 7; visibly poor partial results | Implementation bug made it a speed-head-only run. The checkpoint decoder did not receive LoRA. |

Do not treat a low open-loop validation loss as a closed-loop success.  Model
selection ultimately requires the same 20-mission closed-loop evaluation and,
for claims, matched multi-seed results.

## LoRA module-matching bug and fix

The TF++ checkpoint predictor contains these linear modules:

```text
checkpoint_decoder.encoder
checkpoint_decoder.decoder
```

The v1 regex ended each alternative with `\.` and therefore matched neither
module.  Only these speed layers were trained:

```text
target_speed_network.0
target_speed_network.2
```

The fixed regex is:

```text
^checkpoint_decoder\.(encoder|decoder)$,^target_speed_network\.
```

Training now also accepts `--lora-require-modules`.  A run fails before its
first optimization step unless all four expected modules are installed.  This
prevents another silent speed-only run.

## Current fixed Stage-1 run

Training run:

```text
train_stage1_tfpp_original_objective_lora_v2_fixed_gpu0
```

- DL2 training completed normally with early stopping.
- Best epoch: 2.
- Best validation `tfpp_original_loss`: 0.76094.
- Validation components: checkpoint 0.16125, target speed 0.59969.
- Verified LoRA modules: checkpoint encoder/decoder and both speed-head linear
  layers (four modules total).

The corresponding 20-mission Tesla + canonical-sensor evaluation was started
on DL2:

```text
eval_stage1_tfpp_original_lora_v2_fixed_target20_5video_20260816
```

At the time of this handoff mission 0 was still running, so no closed-loop
claim can be made yet.  The supervisor is resumable.  The first five missions
record video and the remaining fifteen do not.

DL2 result locations:

```text
/media/aimlab/HDD00/users/byeongjae/workspace/teach2drive/data/runs/train_stage1_tfpp_original_objective_lora_v2_fixed_gpu0
/media/aimlab/HDD00/users/byeongjae/workspace/teach2drive/data/runs/eval_stage1_tfpp_original_lora_v2_fixed_target20_5video_20260816
```

Live status:

```bash
ssh DL2
tail -f /media/aimlab/HDD00/users/byeongjae/workspace/teach2drive/data/runs/eval_stage1_tfpp_original_lora_v2_fixed_target20_5video_20260816/supervisor.log
watch -n 2 'cat /media/aimlab/HDD00/users/byeongjae/workspace/teach2drive/data/runs/eval_stage1_tfpp_original_lora_v2_fixed_target20_5video_20260816/summary.tsv'
```

## DL2 operational notes

- Training image: `teach2drive-adapter:dl2`.
- Evaluation image: `teach2drive-eval-py310:dl2`.
- Training launcher: `configs/run_dl2_train_container.sh`.
- Fixed training recipe:
  `configs/train_stage1_tfpp_original_objective_lora_dl2.sh`.
- Resumable evaluation:
  `scripts/supervise_policy_preserving_adapter_eval_dl2.sh` calling
  `scripts/run_policy_preserving_adapter_eval_dl2.sh`.
- An unattended SSH evaluation must launch CARLA with an empty `DISPLAY` while
  using `-RenderOffScreen`.  `DISPLAY=:0` crashed without matching Xauthority;
  the DL2 runner now explicitly uses `DISPLAY=`.

Checkpoints, datasets, evaluation videos, and run directories are intentionally
not stored in Git.  Another machine needs equivalent CARLA Garage, CARLA 0.9.15,
TF++ weights, and dataset mounts.

## Recommended decision after the current evaluation

1. Finish all 20 missions and compare per-mission PASS, route, composed score,
   and infractions against the matched Tesla canonical baseline.
2. If v2-fixed does not robustly exceed baseline, do not increase LoRA capacity
   blindly.  Compare checkpoint and target-speed outputs at the first divergence
   frame and separate lateral-plan errors from stop/go timing errors.
3. Preserve exact raw TF++ route/target-speed labels and repair hazard sampling;
   do not return to future-realized speed as the target-speed label.
4. Validate any promising Stage 1 over three matched seeds before combining it
   with shifted-camera Stage 2.
