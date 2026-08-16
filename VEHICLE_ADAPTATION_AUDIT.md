# TF++ 차량 전용 적응 설계 감사

작성일: 2026-08-15
범위: canonical RGB/LiDAR 입력을 유지하고 Lincoln MKZ에서 Tesla Model 3로 차량만 바꾸는 경우

## 0. 결론

차량만 바뀐 문제를 3시간 데이터로 푸는 것은 가능성이 충분히 있다. 다만 지금까지의 주요 학습은 실제로는 “Lincoln용 TF++ 정책을 Tesla dynamics에 맞게 옮기는 문제”를 풀지 않고, 3시간 PDM-Lite 데이터로 TF++ planner를 새로 모방하는 문제를 풀었다. 이 과정에서 TF++ 원본과 다른 label까지 사용했다. 그래서 데이터가 부족하다기보다 학습 목표가 어긋난 것이 첫 번째 병목이다.

가장 중요한 발견은 다음 네 가지다.

1. 기존 3시간 원시 데이터에는 TF++가 원래 학습하는 `route checkpoint`와 PDM-Lite의 직접 `target_speed`가 모두 남아 있다. 전체 주행 데이터를 다시 수집할 필요는 없다.
2. 현재 인덱스는 이를 버리고 `0.5/1.0/1.5/2.0초 뒤 실제 ego pose`와 `미래 실제 속도`를 label로 만들었다. 이것은 TF++의 path + target-speed 학습 계약과 다르다.
3. 원시 hazard label은 존재하지만 현재 인덱스에는 traffic light, stop sign, front vehicle, junction yield 표본이 각각 0개로 들어갔다. 따라서 여러 실험에서 설정한 hazard weighting과 hazard loss가 사실상 전부 no-op이었다.
4. 차량 전용 적응의 첫 대상은 perception이나 planner가 아니라 `source vehicle의 motion response를 target vehicle에서 복원하는 plan-to-actuation interface`여야 한다. 현재 dynamics v1/v2가 이 방향에 가장 가까웠지만, source Lincoln response를 직접 목표로 삼지 않았고 lateral만 보정했으며 4 Hz 데이터로 transient dynamics를 식별했다.

따라서 다음 주 실험의 핵심은 또 다른 planner residual을 만드는 것이 아니다. 먼저 원본 TF++와 controller는 고정하고, Lincoln에서 같은 명령이 만든 body-frame motion을 Tesla가 재현하도록 20 Hz vehicle adapter를 식별해야 한다. 그 뒤 남는 실패에만 정확한 TF++ label을 이용한 target-speed/path 보정을 적용하는 것이 맞다.

## 1. 우리가 실제로 풀어야 하는 문제

원본 폐루프는 다음과 같다.

```text
sensor observation o_t
  -> frozen TF++ policy pi(o_t, speed_t, route_t)
  -> checkpoint P_t, target-speed distribution V_t
  -> source-tuned controller C_s(P_t, V_t, state_t)
  -> control u_s
  -> Lincoln dynamics f_s
  -> next state x_(t+1)
```

차량만 Tesla로 바꾸면 `pi`의 weight와 센서 형식은 그대로지만 마지막 dynamics가 `f_t`로 바뀐다. 그러면 속도와 자세가 달라지고, 다음 프레임부터 TF++가 받는 image/LiDAR/current-speed도 달라진다. 작은 control 차이가 planner 입력 분포의 차이로 되먹임되는 구조다.

우리가 원하는 adapter의 직접 목표는 아래 식이다.

```text
u_t = A_vehicle(P_t, V_t, u_s, speed_t, yaw_rate_t, ...)

f_t(x_t, u_t) ~= f_s(x_t, u_s)
```

즉 Tesla가 PDM-Lite의 운전 스타일을 새로 배우는 것이 아니라, 원본 TF++가 기대한 Lincoln의 속도·yaw·lateral response를 재현해야 한다. 추론 시 필요한 것은 기존 sensor, speed, IMU/yaw-rate, base TF++ 출력뿐이므로 sensor-only 제한과도 맞는다.

이 분리는 driving policy가 local trajectory를 출력하고 저수준 controller가 dynamics 차이를 흡수하도록 하라는 modular transfer 연구와 일치한다. [Driving Policy Transfer via Modularity and Abstraction](https://arxiv.org/abs/1804.09364)은 perception-policy-control 사이의 추상화를, [robust-control 기반 policy transfer](https://arxiv.org/abs/1812.03216)는 source policy가 만든 reference trajectory를 target vehicle의 robust controller가 추종하는 구성을 사용한다.

## 2. 지금까지 결과를 같은 의미끼리 다시 분류한 표

| 실험 | 입력/차량 | 결과 | 이번 감사에서의 해석 |
|---|---|---:|---|
| native TF++ | canonical, Lincoln | 14/20, composed 94.36 | 목표 reference. 아직 matched 3-seed는 없음 |
| Tesla baseline seed 0/1/2 | canonical, Tesla | 7/20, 8/19, 9/19 | robust 기준은 단발 7이 아니라 평균 약 8 pass |
| historical v4 | shifted, Tesla | 10/20, composed 70.5 | paired RGB feature alignment의 부분 성공. vehicle-only 증거는 아님 |
| v6 vehicle LoRA 4종 | canonical, Tesla | 0, 1, 0, 0 pass 수준 | 3시간 expert imitation으로 pretrained policy를 크게 훼손 |
| controller scalar/system-ID | canonical, Tesla | 평균 pass 소폭 변화, composed 무개선 | 단일 gain은 부족했고 일부 parameter는 실제 control lever가 아니었음 |
| dynamics inverse v1 | canonical, Tesla | 단발 10/20, composed 83.74 | vehicle dynamics 방향의 첫 유의미한 신호지만 single-run |
| dynamics v2 | canonical, Tesla | 3-seed pass 9/8/10, composed 86.54/79.97/79.32 | 평균 약 9 pass. 작지만 현재 가장 정합적인 partial success |
| full PDM oracle + TF++ PID | canonical, Tesla | 18/20, composed 99.19 | Tesla에서 좋은 motion이 물리적으로 가능하다는 진단. 배포 가능한 sensor policy는 아님 |
| PDM checkpoint + TF++ speed | canonical, Tesla | 12/20 | checkpoint도 기여한다는 ablation |
| TF++ checkpoint + PDM speed | canonical, Tesla | 14/20 | longitudinal decision의 영향이 큼을 시사. mixed harness sanity check 전까지 정밀 attribution은 잠정 |
| 최근 checkpoint residual | canonical, Tesla | 8/20, composed 72.99 | baseline 평균을 넘지 못함. future ego path label 사용 |
| 최근 speed residual | canonical, Tesla | 5/14 진행 중 | 아직 미완료이며 future realized speed label 사용 |

여기서 10/20은 두 종류 모두 단발 결과다. historical v4는 shifted sensor이고 dynamics v1도 single-run이다. 현재 sensor-only canonical 차량 적응에서 재현된 최고 근거는 dynamics v2의 평균 약 9 pass이며, 아직 native 14에 근접했다고 말할 수 없다.

또 native는 1회만 있고 Tesla는 3회가 있으므로 “정확히 7 pass 차이”도 아직 통계적으로 확정된 값이 아니다. 동일 executable, traffic seed, container, sensor 설정으로 Lincoln/Tesla 각각 3-seed를 먼저 맞춰야 한다.

## 3. TF++가 실제로 학습하고 실행되는 방식

현재 사용하는 checkpoint는 `carla_garage` 공식 구현의 3-model ensemble이다. 각 모델의 흐름은 다음과 같다.

```text
RGB RegNet + LiDAR-BEV RegNet
  -> 4단계 multi-scale transformer fusion
  -> BEV spatial tokens
  -> 6-layer transformer decoder
  -> 10 spatial checkpoints + 8-class target-speed logits
  -> probability-weighted target speed
  -> dynamic-lookahead lateral PID + longitudinal regression controller
  -> steer / throttle / brake
```

중요한 학습 계약은 다음과 같다.

- checkpoint는 미래 시간의 ego 위치가 아니다. 첫 점이 차량 중심에서 약 2.5 m, 이후 1 m 간격인 10개의 공간 경로점이다.
- target speed는 미래 실제 속도가 아니다. PDM-Lite가 현재 장면에서 결정한 직접 목표 속도다.
- PDM-Lite의 연속 target speed는 `[0, 4, 8, 10, 13.89, 16, 17.78, 20] m/s`에 two-hot encoding된다.
- 추론은 현재 설정 `UNCERTAINTY_WEIGHT=1`에 따라 argmax가 아니라 class probability의 가중평균을 사용한다.
- CARLA는 20 Hz지만 학습 데이터 저장은 매 5 tick이므로 effective 4 Hz다. 따라서 TF++ planner label 학습에서 4 Hz 자체는 오류가 아니다.
- 반면 PID memory, steering delay, braking transient와 같은 vehicle dynamics 식별에는 4 Hz가 거칠다. 여기서는 20 Hz가 필요하다.

[TF++ 논문](https://arxiv.org/html/2306.07957)은 weighted target speed가 argmax보다 vehicle collision을 줄였고, perception pretraining 뒤 전체 loss로 end-to-end 최적화한 모델이 backbone을 고정한 모델보다 10 DS 높았다고 보고한다. 또한 185k에서 555k frame으로 늘릴 때 6 DS가 개선됐다. 이는 40k frame으로 TF++의 planner를 다시 학습하면서 backbone 전체를 고정하거나 작은 residual 하나만 붙이는 방식에 분명한 ceiling이 있음을 보여준다.

[PDM-Lite TF++ 데이터 논문](https://arxiv.org/html/2412.09602)은 337k frame을 기본으로 쓰고, 최종 모델은 531k frame을 사용한다. label도 명시적으로 path checkpoint, expert target speed, auxiliary perception label이다. 우리의 40k frame은 pretrained policy의 저차원 vehicle interface를 적응하기에는 충분할 수 있지만, perception/planning policy를 다시 배우기에는 원본 규모의 약 8~12%에 불과하다.

## 4. 데이터·학습 pipeline에서 확인된 구체적인 문제

### 4.1 원본 label을 다른 문제로 변환했다

원시 `measurements/*.json.gz`에는 모든 frame에 `route`와 `target_speed`가 존재한다. 예시 route는 이미 약 2.5 m에서 시작해 1 m 간격으로 저장돼 있다.

하지만 현재 token index의 정의는 다음과 같다.

- `traj_targets`: 0.5/1.0/1.5/2.0초 뒤 실제 ego `[dx, dy, dyaw]`
- `speed_targets`: 같은 horizon의 미래 실제 전진 속도

직접 PDM target speed와 미래 실제 속도의 평균 absolute difference는 다음과 같다.

| horizon | MAE | correlation | stop/go 불일치율 |
|---:|---:|---:|---:|
| 0.5 s | 1.20 m/s | 0.836 | 6.4% |
| 1.0 s | 1.52 m/s | 0.814 | 9.1% |
| 1.5 s | 1.81 m/s | 0.774 | 11.7% |
| 2.0 s | 2.12 m/s | 0.724 | 13.6% |

특히 PDM target speed가 0인 frame에서도 향후 속도 평균은 `[0.30, 0.39, 0.56, 0.77] m/s`다. 제동 직후의 관성이나 2초 이내 release 때문에 미래 실제 속도는 0이 아니다. 이 label로 speed residual을 학습하면 “지금 brake해야 한다”는 decision 대신 “차량이 나중에 실제로 얼마나 움직였는가”를 배우게 된다.

최근 checkpoint residual도 `ROUTE_TARGET_SOURCE=future_ego_path`라서 원본 TF++ route가 아니라 전문가가 실제로 주행한 미래 궤적을 공간 재표본화했다. 6초 안에 11.5 m를 주행하지 못한 frame은 무효가 되어 validation의 `route_valid_ratio`가 0.576이었다. 즉 path supervision의 42.4%가 사라졌다.

### 4.2 hazard 정보가 학습 인덱스에서 사라졌다

원시 41,473 frame의 hazard 비율은 다음과 같다.

- vehicle hazard: 12.5%
- traffic-light hazard: 8.7%
- walker hazard: 0.7%
- stop-sign hazard: 3.1%
- 위 항목 중 하나 이상: 24.3%

그런데 현재 두 index의 stop-reason 분포는 다음과 같다.

- traffic_light: 0
- stop_sign: 0
- front_vehicle: 0
- junction_yield: 0
- 대부분 none, unknown_stop, route_end

converter가 `frame["hazard"]`를 기록했지만 기존 token builder는 `traffic_light`, `stop_sign`, `front_vehicle`, `lane.is_junction`이라는 다른 schema만 읽는다. 그래서 최근 학습 summary에서도 `hazard_ratio=0`, 모든 hazard-specific recall count가 0이다. 설정 파일에서 hazard sample weight와 loss weight를 아무리 바꿔도 실제 gradient에는 들어가지 않았다. 남은 실패가 교차로 vehicle collision 중심인 상황에서 매우 큰 결함이다.

### 4.3 3시간이라는 시간보다 frame 구성의 편향이 더 크다

원시 데이터 감사 결과는 다음과 같다.

- 41,473 frame, 313개 결과 route, 약 2.88시간
- PDM target speed 0: 44.9%
- 실제 speed < 0.2 m/s: 46.8%
- Perfect route에서 나온 frame: 33.8%, 약 0.97시간
- min-speed infraction route에서 나온 frame: 50.9%
- timeout 2 route가 전체 frame의 8.1%를 차지하며 그중 target-speed 0 비율은 96% 이상
- 상위 8개 긴 episode가 전체 frame의 19.1%를 차지

데이터가 완전히 못 쓸 정도라는 뜻은 아니다. 다만 uniform frame sampling은 “다양한 3시간”이 아니라 장시간 정지한 몇 경로를 과대표집한다.

PDM-Lite TF++ 논문 방식대로 target speed가 0.1 m/s 이상 변하거나 checkpoint angle이 0.5도 이상 변한 frame을 모두 유지하고, 나머지 중 14%만 유지하면 현재 데이터의 약 49.6%가 남는다. 이는 약 20.5k개의 정보량 높은 frame이며, 정지 대기 frame을 줄이면서 제동 진입·release·turn 변화는 보존한다. speed class frequency weighting 대신 이 방식이 맞다.

### 4.4 vehicle adaptation과 expert-policy replacement가 섞였다

PDM-Lite expert trajectory와 TF++ trajectory는 둘 다 합리적이어도 출발, 양보, turn 진입 시점이 다를 수 있다. target vehicle 데이터의 expert pose/control을 전 frame에서 강하게 모방하면 차량 차이를 보정하는 것이 아니라 pretrained TF++의 정책 스타일을 PDM style로 바꾼다. v6에서 imitation을 강하게 할수록 더 붕괴한 결과가 이 해석과 일치한다.

Oracle 18/20은 “PDM plan과 speed를 주면 Tesla + TF++ PID 조합으로 이 benchmark를 풀 수 있다”는 feasibility test다. 그것이 곧 PDM 출력을 3시간 데이터로 그대로 behavior-clone하는 것이 최선이라는 뜻은 아니다. 우리의 1차 목표가 native TF++의 14/20을 복원하는 것이라면, reference는 우선 Lincoln에서 실행된 TF++의 closed-loop motion이어야 한다.

### 4.5 open-loop validation이 선택 기준을 대신했다

낮은 train/validation loss와 closed-loop pass 사이에는 일관된 상관이 없었다. 최근 checkpoint residual은 validation loss 0.0716, route lateral loss 0.0495로 낮지만 closed-loop는 8/20, composed 72.99였다. speed residual은 validation에서 gate가 0.9987까지 포화됐고 평균 speed residual norm도 0.286이었다. “작은 bounded residual”로 시작했지만 실제로는 거의 모든 sample에 보정을 켠 셈이다.

TF++ 원 논문도 CARLA variance 때문에 각 설정을 3 training seed × 3 evaluation으로 측정했다. 20 mission single-run에서 best checkpoint를 고르면 7과 10의 차이도 쉽게 과대해석된다.

### 4.6 sensor 쪽 과거 실험의 교훈

LiDAR는 sensor 위치에서 들어온 point cloud를 `lidar_to_ego_coordinate()`로 이미 vehicle-origin 좌표계로 변환한 뒤 BEV histogram으로 만든다. 따라서 shifted LiDAR에 추가적인 단순 BEV translation을 적용하면 좌표를 이중 보정할 수 있다. mount 차이는 좌표 원점보다 visibility, self-occlusion, point-density distribution 차이로 남는다.

SE(3) 정보는 이후 sensor stage에서 camera projection/feature alignment 조건으로 쓰는 것이 맞다. vehicle-only canonical stage에서는 extrinsic adapter를 켜지 않아야 한다.

## 5. 실험별 교훈

### 실패로 확정할 수 있는 것

- canonical 입력에서 3시간 PDM future trajectory/speed를 이용한 broad LoRA fine-tuning
- expert pose/control을 pretrained TF++ policy 전체의 대체 정답처럼 쓰는 방식
- steering gain 한 개 또는 단순 PID gain만으로 전체 vehicle gap이 닫힌다는 가설
- open-loop loss나 lateral error만으로 checkpoint를 선택하는 방식
- hazard label이 들어간다고 가정하고 loss weight만 조정한 실험
- sensor와 vehicle 축을 동시에 바꾸고 결과를 vehicle adaptation으로 해석하는 방식

### 부분 성공으로 남겨야 하는 것

- paired canonical RGB teacher를 이용한 v4 feature alignment: sensor stage에 유효한 단서
- dynamics inverse v1/v2: target 차량의 yaw response를 식별하고 bounded correction을 주는 방향은 평균 pass를 약 8에서 약 9로 올린 신호가 있음
- full oracle 18/20: Tesla 물리 및 원본 PID 자체가 절대적 병목은 아니며, plan/speed interface를 통해 높은 성능이 가능함
- speed-only oracle 14/20: 남은 충돌에서 longitudinal decision/timing이 중요함

### 아직 검증되지 않은 것

- Lincoln native의 matched 3-seed 평균
- mixed oracle agent에 checkpoint와 speed를 동시에 넣었을 때 full-oracle agent의 18/20을 재현하는지
- 같은 source command에 대한 Lincoln/Tesla 20 Hz motion-response 차이
- 실제 Tesla bounding box, axle/reference origin, actuator delay를 반영하면 얼마나 gap이 줄어드는지
- direct PDM target-speed label과 raw route label을 정확히 쓰면 head adaptation이 개선되는지

## 6. 권장 복원 설계

### R0. 평가와 trace 기준선 고정

먼저 학습 없이 아래를 고정한다.

1. Lincoln canonical base와 Tesla canonical base를 동일 agent/container/traffic seed로 각각 3회 평가한다.
2. 20 Hz로 `P_t`, target-speed probability, weighted target speed, base control, actual control, speed, yaw-rate, pose, collision/route event를 저장한다.
3. gap mission 2, 3, 5, 8, 10, 13, 15, 17, 18에서 최초 divergence 시점을 찾는다.
4. 두 oracle 출력 동시 주입 sanity check가 18/20을 재현하는지 확인한다.

이 단계의 목적은 실패를 세 종류로 나누는 것이다.

- plan은 같은데 target vehicle motion만 다름: dynamics/controller 문제
- motion을 맞추기 전부터 P/V가 다름: sensor/input/config 문제
- motion은 맞지만 Tesla만 collision: vehicle footprint/reference-origin 문제

### R1. Source-reference vehicle adapter

원본 TF++ weight와 `P,V`는 완전히 고정한다. Lincoln용 controller가 만든 base action도 그대로 계산한다. 작은 adapter가 Tesla control만 변환한다.

```text
frozen TF++ P,V
  -> frozen original controller C_s
  -> base steer/throttle/brake u_s
  -> A_vehicle(u_s, speed, yaw-rate, recent control/state history)
  -> Tesla steer/throttle/brake u_t
```

학습 목표는 expert steer imitation이 아니라 source response imitation이다.

```text
L_motion =
  w_yaw * |yaw_rate_target - yaw_rate_source_reference|
  + w_speed * |accel_target - accel_source_reference|
  + w_pose * body-frame pose error over 0.25~1.0 s
  + smoothness + identity/bound penalties
```

여기서 source reference는 Lincoln의 동일 excitation/TF++ rollout에서 얻는다. 추론 때 Lincoln 데이터나 oracle은 필요 없다.

권장 adapter는 거대한 neural controller가 아니라 다음 저차원 구성이다.

- lateral: speed-conditioned monotonic inverse steering map + steering lag state
- longitudinal: target acceleration 기반 throttle inverse map, brake deceleration/lag map
- shared: 0.5~1초 causal history, saturation/dead-zone, bounded residual
- optional: disturbance observer 또는 작은 residual state-space model

현재 dynamics v2와 다른 점은 `expert future pose curvature`가 아니라 `Lincoln motion response`를 정답으로 쓰고, lateral뿐 아니라 longitudinal까지 같이 맞춘다는 점이다.

기존 4 Hz PDM data로 quasi-static initialization은 가능하지만 actuator/PID transient를 위해 짧은 20 Hz calibration을 새로 따는 것이 좋다. 전체 route dataset을 다시 딸 필요는 없고, 각 차량에서 직선 가감속, step/sine steering, 복합 turn을 포함한 10~30분이면 저차원 system-ID에는 훨씬 유용하다.

R1 통과 조건은 selected mission pass가 아니라 먼저 motion response다.

- held-out command sequence의 0.5초 yaw/speed prediction error 50% 이상 감소
- source-reference lateral/heading/speed trace error 50% 이상 감소
- control saturation과 brake false-positive 증가 없음
- 그 뒤 matched 20 missions × 3 seeds에서 baseline보다 pairwise 개선

### R2. 정확한 longitudinal head 보정

R1 뒤에도 wrong brake/go decision이 남을 때만 수행한다.

기존 raw data로 새 index를 만든다.

- label: `measurement["target_speed"]`의 TF++ two-hot encoding
- hazard: raw `light_hazard`, `stop_sign_hazard`, `vehicle_hazard`, `walker_hazard`
- normal frame: frozen TF++ logits distillation을 강하게 적용
- target-speed 변화/hazard 진입·해제 frame: PDM direct target-speed supervision 적용
- timeout/blocked route 제외, 긴 정지 sequence cap
- speed class inverse-frequency weighting 사용하지 않음

첫 모델은 target-speed head의 작은 residual이나 calibration layer로 제한한다. gate가 항상 켜지지 않도록 L0/L1-style activation penalty와 per-event gate metric을 둔다. normal frame에서 base output drift가 기준치를 넘으면 폐기한다.

PDM target speed는 training-only privileged label이다. 이는 TF++ 원 학습과 같은 방식이고 추론은 여전히 sensor-only다. 다만 연구 제약이 “일반 주행 log에 존재하는 값만 label로 허용”이라면 직접 사용할 수 없고, 그 경우 behavior에서 decision을 역추정해야 하므로 성능 ceiling이 낮아진다.

### R3. 정확한 path 보정

R1/R2 뒤 trace에서 checkpoint geometry가 실제로 틀린 frame에만 적용한다.

- label: raw `measurement["route"]`, 첫 2.5 m + 이후 1 m 간격 10점
- future ego pose를 path label로 사용하지 않음
- turn/hazard transition frame에 집중
- straight/normal frame에서는 base checkpoint identity/distillation
- 처음에는 lateral residual만 허용하고 x/progress는 고정

단순 output residual이 부족하면 마지막 checkpoint decoder와 마지막 1~2개 transformer-decoder layer만 낮은 LR로 푼다. v6 LoRA 실패가 “모든 PEFT가 불가능”하다는 뜻은 아니지만, 정확한 label과 source replay 없이 다시 LoRA부터 시작해서는 안 된다.

TF++ 논문에서 backbone을 완전히 고정한 second stage가 낮았다는 결과도 고려해야 한다. 다만 40k target data로 perception 전체를 풀면 catastrophic forgetting 위험이 크므로, source replay 또는 frozen-base distillation을 함께 쓸 수 있을 때만 점진적으로 unfreeze한다.

### R4. Sensor 위치 adapter는 마지막에 결합

canonical Tesla가 안정적으로 native reference에 근접한 뒤에만 shifted RGB/LiDAR stage를 연다.

```text
shifted sensor
  -> SE(3)-conditioned camera/feature canonicalizer
  -> frozen vehicle-adapted TF++ policy from R1~R3
```

paired canonical/shifted RGB는 같은 frame이므로 feature alignment에 매우 좋은 데이터다. historical v4의 유일한 부분 성공도 이 축에서 나왔다. LiDAR는 이미 ego coordinate로 변환되므로 단순 translation보다 visibility/density residual만 다뤄야 한다.

## 7. 데이터셋을 다시 따야 하는가

전체 3시간 PDM-Lite dataset은 다시 딸 필요가 없다. 원시 파일에 다음 정보가 이미 있다.

- canonical RGB (`rgb_canonical`)
- shifted RGB
- ego-coordinate LiDAR point cloud
- raw spatial route checkpoints
- direct PDM target speed
- brake/control 및 차량 pose
- traffic-light, vehicle, walker, stop-sign hazard
- route 결과와 infraction

필요한 것은 재수집보다 `lossless re-indexing`이다. converter가 아래 필드를 버리지 않도록 바꿔야 한다.

```text
source_frame_index
route / route_original
target_speed
all hazard flags
result status / infractions
actual camera and LiDAR layout metadata
```

추가 수집이 필요한 것은 두 종류뿐이다.

1. 저수준 vehicle dynamics용 20 Hz calibration 10~30분/차량
2. offline expert-only data로도 복구가 안 되는 경우, Tesla TF++가 실제로 방문한 실패 state에서 teacher/human correction을 받은 소량의 intervention data

두 번째가 필요한 이유는 imitation learning의 covariate shift다. expert가 방문한 정상 state만 학습하면 student가 실수해 들어간 state의 복구 행동은 데이터에 없다. [DAgger](https://arxiv.org/abs/1011.0686)는 learner가 유도한 observation distribution에서 label을 모으는 이유를 이론화했고, 최근 driving world-model 연구도 expert-only behavior cloning이 closed-loop에서 drift/collision으로 이어짐을 보인다. [Mitigating Covariate Shift in Imitation Learning for Autonomous Vehicles](https://arxiv.org/abs/2409.16663)

실차 적용 가능성을 지키려면 이 correction은 추론 때 쓰는 closed-loop oracle이 아니라 training-time human takeover, safety driver annotation, 또는 offline learned dynamics rollout으로 구현할 수 있다.

## 8. 현실적인 기대치

14/20 근처 복원은 불가능한 목표로 보이지 않는다. 같은 sensor policy가 이미 Lincoln에서 14/20이고, Tesla에서도 PDM plan/speed + 원본 PID로 18/20이 가능했다. 따라서 target 차량이 해당 benchmark를 물리적으로 수행하지 못하는 것은 아니다.

다만 “기존 3시간 expert log만 그대로 behavior-clone하면 자동으로 14/20이 된다”는 기대는 근거가 약하다.

- 원본 TF++는 수십만 frame으로 policy를 학습했다.
- 현재 3시간은 40k frame이며 정지/timeout 편향이 크다.
- 차량 차이가 만든 student-state 분포가 expert log에는 없다.
- benchmark 20 mission은 단발 variance가 크다.

3시간이 충분한 범위는 저차원 vehicle dynamics/interface 적응이다. 3시간으로 고차원 perception/planner를 재학습하려 한 것이 오래 걸린 핵심 이유다.

## 9. 바로 실행할 우선순위

1. 진행 중인 speed residual 평가는 결과 기록만 하고, 같은 recipe의 추가 tuning은 중단한다.
2. raw measurement를 보존하는 v2 index를 만든다. direct route, direct target speed, hazard, route-quality flag를 unit test한다.
3. Lincoln/Tesla matched 3-seed baseline과 20 Hz trace를 만든다.
4. mixed oracle both-output sanity check를 한다.
5. source-reference lateral + longitudinal vehicle adapter R1을 먼저 학습한다.
6. R1 후 남은 failure trace에 따라 R2 speed 또는 R3 path 중 하나만 연다.
7. canonical vehicle-only가 안정된 뒤 SE(3) sensor stage를 다시 결합한다.

이 순서를 지키면 각 실험이 “차량 dynamics를 고쳤는가, longitudinal decision을 고쳤는가, path를 고쳤는가, sensor를 고쳤는가” 중 하나만 답하게 된다. 지금까지처럼 한 결과에서 여러 원인을 동시에 추측하는 병목을 끊을 수 있다.
