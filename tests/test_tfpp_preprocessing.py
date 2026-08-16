import unittest
from types import SimpleNamespace

import numpy as np
import torch

from teach2drive_adapter.tfpp_preprocessing import lidar_to_tfpp_histogram
from teach2drive_adapter.transfuserpp_bridge import camera_to_transfuserpp_rgb, lidar_to_transfuserpp_bev


class TFPPPreprocessingTest(unittest.TestCase):
    def test_lidar_matches_carla_garage_histogram_operations(self):
        points = np.array(
            [
                [-31.9, -31.9, 0.3],
                [-31.9, -31.9, 0.4],
                [0.0, 0.0, 0.21],
                [0.0, 0.0, 0.1],
                [31.9, 31.9, 0.5],
                [31.9, 31.9, 101.0],
            ],
            dtype=np.float64,
        )

        actual = lidar_to_tfpp_histogram(points)

        xbins = np.linspace(-32, 32, 257)
        ybins = np.linspace(-32, 32, 257)
        above = points[(points[:, 2] < 100.0) & (points[:, 2] > 0.2)]
        histogram = np.histogramdd(above[:, :2], bins=(xbins, ybins))[0]
        histogram[histogram > 5] = 5
        expected = (histogram / 5).T[None].astype(np.float32)

        self.assertEqual(actual.shape, (1, 256, 256))
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_rgb_bridge_is_exact_top_crop_without_resize(self):
        pixels = torch.arange(3 * 512 * 1024, dtype=torch.int64).remainder(256).to(torch.float32)
        camera = (pixels.reshape(1, 1, 3, 512, 1024) / 255.0).contiguous()
        config = SimpleNamespace(
            camera_height=512,
            camera_width=1024,
            crop_image=True,
            cropped_height=384,
            cropped_width=1024,
            camera_fov=110.0,
        )

        actual = camera_to_transfuserpp_rgb(camera, ["front"], config)
        expected = camera[:, 0, :, :384, :] * 255.0
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_bridge_rejects_legacy_sensor_shapes(self):
        rgb_config = SimpleNamespace(camera_height=512, camera_width=1024)
        with self.assertRaisesRegex(ValueError, "original camera resolution"):
            camera_to_transfuserpp_rgb(torch.zeros(1, 1, 3, 360, 640), ["front"], rgb_config)

        lidar_config = SimpleNamespace(
            lidar_seq_len=1,
            use_ground_plane=False,
            lidar_resolution_height=256,
            lidar_resolution_width=256,
        )
        with self.assertRaisesRegex(ValueError, "legacy occupancy/height/intensity"):
            lidar_to_transfuserpp_bev(torch.zeros(1, 3, 128, 128), lidar_config)


if __name__ == "__main__":
    unittest.main()
