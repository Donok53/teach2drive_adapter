# Tesla canonical vehicle adaptation: 2026-08-19 iteration

## Fixed evaluation contract

- Track: sensor-only (`SENSORS`), no privileged future state at deployment.
- Ego vehicle: `vehicle.tesla.model3`.
- Sensor rig: released TF++ canonical RGB and LiDAR (`tfpp_ego`).
- TF++ ensemble order: `model_0030_1.pth`, `model_0030_0.pth`, `model_0030_2.pth`.
- Primary result: 20 fixed missions; final candidate records the first five videos.
- A PASS is `Perfect` with no infraction. Even a high composed score is a FAIL.

Reference outcomes:

- Released TF++ + Lincoln: 14/20.
- Released TF++ + Tesla + canonical sensors: about 7/20.
- Privileged PDM longitudinal oracle + TF++ PID: 18/20. This is diagnostic only.

## V20: one-second vehicle-hazard lead target

Run: `train_stage1a_speed_runtime_v20_targetquery_vehiclelead4_gpu0_20260819`.

The frozen TF++ ensemble is unchanged. A post-ensemble residual receives the
released target-speed-head context, checkpoints, and base speed logits. It can
change only target-speed logits, and deployment uses a hard gate.

Training used four lead frames (the paired corpus is 4 Hz), hazard sampling
weight 6, and positive gate BCE weight 3. The best offline validation epoch was
epoch 3 (`tfpp_speed_policy_loss=1.335392`). Closed-loop evidence disagreed:

- Epoch 2, residual scale 3.0: mission 003 PASS and mission 013 PASS.
- The same candidate regressed mission 002 from base PASS to FAIL and changed
  20--50% of frames in several routes.
- Mission 012 avoided the base collision, but repeated slowing caused a
  minimum-speed infraction (`97.144`, still FAIL).
- Epoch 2 and epoch 3 both failed mission 010 with a vehicle collision (`60`).
- Epoch 2 scale 2.25 failed mission 003 with timeout + collision at 52.01%
  route (`31.206`). Scale 1.5 had also failed; scale 3.0 was a narrow result.

Conclusion: V20 is not a full-evaluation candidate. Offline validation loss and
single-route improvement concealed excessive intervention.

### Data audit

`sample_frame_indices` are contiguous row indices within each `frames.jsonl`,
so the lead-label indexing is aligned. Across all 41,473 paired frames:

- Current `vehicle_hazard`: 5,184 frames (12.50%).
- Four-frame lead mask: 7,416 frames (17.88%).
- 99.75% of current vehicle-hazard frames are expert brake / 0 m/s.
- In the lead-4 mask, 75.36% are already brake; 24.64% are anticipatory frames.

The main error was distribution weighting, not an index shift: 6x hazard
sampling plus 3x positive BCE made the training stream far more brake-heavy
than deployment. Train gate false-positive remained roughly 18--27%.

## Dynamics composition ablations

V20 epoch 2 scale 3.0 plus the previously trained bounded Tesla yaw dynamics:

- Restored mission 002 to PASS.
- Regressed the speed-only mission 003 PASS to timeout + collision at 51.48%
  route (`30.888`).

Sparse V4 speed residual plus the same dynamics:

- Mission 002 scored `99.823` but was still FAIL due to a minimum-speed
  infraction (average speed 99.41% of surrounding traffic).

Conclusion: neither composition is a candidate. Vehicle dynamics can repair a
lateral regression, but applying it throughout the route changes scenario
timing enough to destroy another required PASS.

## V21: conservative lead-4 gate

Run: `train_stage1a_speed_runtime_v21_conservative_lead4_gpu0_20260819`.

Only the intervention prior changed relative to V20:

- vehicle-hazard sample weight: 6.0 -> 1.0;
- positive gate BCE weight: 3.0 -> 1.0;
- gate bias: 0.0 -> -3.0;
- prior KL: 2.5 -> 5.0;
- non-correction residual norm: 0.25 -> 0.50.

TF++ remains fully frozen and lead time remains four frames, isolating whether
the closed-loop regression came from over-intervention. Runtime screening uses
a hard gate of 0.8 first, because this benchmark marks even small average-speed
changes as an infraction. The first sentinel order is mission 002 (preserve),
003 (improve), 010 and 012 (collision safety).

Status: training/evaluation in progress.

## DL2 evaluation GPU mapping audit

The DL2 runner previously launched CARLA with both
`CUDA_VISIBLE_DEVICES=${GPU}` and `-graphicsadapter=${GPU}`.  Direct process
inspection with `nvidia-smi pmon` showed that an evaluation declared as GPU1
actually placed CARLA compute/render work on physical GPU0.  When GPU0 training
was active, a 500-second wall timeout advanced only about 50 seconds of
simulation.  V39 therefore ended at 53.58% route because of evaluator resource
contention and is not valid policy evidence.

DL2's Unreal/Vulkan enumeration is reversed relative to the NVIDIA index used
by the evaluator container.  The corrected contract is:

- evaluator container: `--gpus device=1`;
- CARLA: all physical devices visible, `-graphicsadapter=0`;
- validation: CARLA is observed as active `C+G` work on physical GPU1 before a
  policy result is accepted.

All post-fix runs use new run directories; interrupted pre-fix runs are never
resumed or merged into summaries.

Post-fix identity checks also exposed scenario variance.  V21 epoch 1 with a
0.8 hard gate made zero interventions on mission 001, yet that repetition
collided with a Ford Mustang and scored 60; an earlier exact-base repetition
had passed.  A single route flip is therefore not sufficient attribution.

In the isolated post-fix V21 epoch-1 screen:

- mission 002: PASS 100, zero interventions at threshold 0.8;
- mission 003: route 100 but vehicle collision, score 60, zero interventions
  at thresholds 0.8 and 0.5.

Offline replay of the same traces shows that threshold 0.2 would make only six
meaningful target-speed changes on mission 002 and five on mission 003.  That
candidate is the next closed-loop A/B; it is materially narrower than V20's
hundreds of modified frames.

The isolated threshold-0.2 A/B preserved mission 002 but mission 003 still
collided with the same Dodge Charger at essentially the same world position.
V21 epoch 1 is therefore not a candidate.

## Sensor-only predicted-box stop extension

Run `diag_v49_predbox_base_isolated_m3_20260819` wrapped the released TF++
policy without changing its action and recorded the existing CenterNet BEV
box-head output.  Mission 003 again collided with a Dodge Charger at
`(6773.228, 4921.471)`.  The detector was not the bottleneck:

- the first oncoming car was visible from about 31 m and passed while TF++
  held a zero target speed;
- the second oncoming car was also visible from about 31 m;
- while that car closed from 26 m to 8 m, TF++ released its target speed from
  roughly 3.2 to 10.0 m/s and accelerated into the collision.

This motivates a narrow sensor-only shield using TF++'s own class-0/class-4
boxes.  It does not replace the planner and does not access CARLA actors.
Early closed-loop versions exposed two important negative controls:

- remembering left-turn intent for 40 frames caused a later straight-road
  intervention;
- interpreting PID `brake=1` as a semantic stop was wrong, because PID also
  brakes when current speed is merely above a nonzero target speed.

The accepted trigger under evaluation is consequently limited to:

1. TF++ is currently steering left;
2. its own detector sees a high-confidence oncoming vehicle in the conflict
   lane;
3. TF++ itself has selected a target speed at or below 0.5 m/s.

After triggering, the shield only extends that already-selected stop until
the detected conflict clears (plus eight frames for detector flicker).  In
closed loop, `diag_v52_targetzero_stop_pair_m2_m3_20260819` produced:

- mission 002: zero interventions over the complete trace.  This particular
  repetition still failed with the same base-policy lane/min-speed variance,
  so it is an action-identity control rather than a PASS claim;
- mission 003: PASS 100 with no infraction.  The shield triggered once,
  extended the existing stop for 83 frames, released after the second
  oncoming car cleared, and avoided the repeated Dodge Charger collision.

The next cumulative test is
`eval_v53_v1_targetspeed_boxshield_selected_20260819`: combine this shield with
the prior sensor-only target-speed adapter that scored 11/20, then screen
mission 003 and 013 improvements against mission 004 and 005 preservation.
