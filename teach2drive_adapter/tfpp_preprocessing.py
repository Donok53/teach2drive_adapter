"""Exact sensor preprocessing used by the pretrained CARLA Garage TF++ model.

Keep this module NumPy-only so offline dataset conversion can reuse the TF++
input contract without importing CARLA or the full CARLA Garage training stack.
"""

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class TFPPHistogramConfig:
    """LiDAR histogram constants from ``carla_garage/team_code/config.py``."""

    min_x: int = -32
    max_x: int = 32
    min_y: int = -32
    max_y: int = 32
    pixels_per_meter: int = 4
    hist_max_per_pixel: int = 5
    lidar_split_height: float = 0.2
    max_height_lidar: float = 100.0
    use_ground_plane: bool = False

    @property
    def height(self) -> int:
        return (self.max_x - self.min_x) * self.pixels_per_meter

    @property
    def width(self) -> int:
        return (self.max_y - self.min_y) * self.pixels_per_meter

    @property
    def channels(self) -> int:
        return 2 if self.use_ground_plane else 1

    def metadata(self) -> Dict[str, object]:
        result = asdict(self)
        result.update(
            {
                "representation": "carla_garage_tfpp_density_histogram",
                "height": self.height,
                "width": self.width,
                "channels": self.channels,
                "dtype": "float32",
            }
        )
        return result


TFPP_HISTOGRAM_CONFIG = TFPPHistogramConfig()


def lidar_to_tfpp_histogram(
    lidar: np.ndarray,
    config: TFPPHistogramConfig = TFPP_HISTOGRAM_CONFIG,
) -> np.ndarray:
    """Convert an ego-frame point cloud to the original TF++ density histogram.

    The operations intentionally mirror
    ``CARLA_Data.lidar_to_histogram_features`` in CARLA Garage. The pretrained
    configuration uses only the above-ground channel.
    """

    lidar = np.asarray(lidar)
    if lidar.ndim != 2 or lidar.shape[1] < 3:
        raise ValueError(f"Expected LiDAR points with shape [N,>=3], got {lidar.shape}")
    lidar = lidar[:, :3]

    def splat_points(point_cloud: np.ndarray) -> np.ndarray:
        xbins = np.linspace(
            config.min_x,
            config.max_x,
            (config.max_x - config.min_x) * int(config.pixels_per_meter) + 1,
        )
        ybins = np.linspace(
            config.min_y,
            config.max_y,
            (config.max_y - config.min_y) * int(config.pixels_per_meter) + 1,
        )
        histogram = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
        histogram[histogram > config.hist_max_per_pixel] = config.hist_max_per_pixel
        return (histogram / config.hist_max_per_pixel).T

    lidar = lidar[lidar[..., 2] < config.max_height_lidar]
    below = lidar[lidar[..., 2] <= config.lidar_split_height]
    above = lidar[lidar[..., 2] > config.lidar_split_height]
    below_features = splat_points(below)
    above_features = splat_points(above)
    if config.use_ground_plane:
        features = np.stack([below_features, above_features], axis=-1)
    else:
        features = np.stack([above_features], axis=-1)
    return np.transpose(features, (2, 0, 1)).astype(np.float32)
