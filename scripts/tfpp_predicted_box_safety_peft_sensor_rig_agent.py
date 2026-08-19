#!/usr/bin/env python3
"""Policy-preserving TF++ adapter followed by the sensor-only box shield."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tfpp_feature_then_fusion_peft_adapter_sensor_rig_agent import (  # noqa: E402
    FeatureThenFusionPeftAdapterSensorRigAgent,
)
from tfpp_predicted_box_safety_sensor_rig_agent import (  # noqa: E402
    PredictedBoxSafetyMixin,
)


def get_entry_point() -> str:
    return "PredictedBoxSafetyPeftSensorRigAgent"


class PredictedBoxSafetyPeftSensorRigAgent(
    PredictedBoxSafetyMixin,
    FeatureThenFusionPeftAdapterSensorRigAgent,
):
    """Run the learned target-speed residual, then apply the narrow shield."""
