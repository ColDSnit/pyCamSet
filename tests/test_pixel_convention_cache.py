"""
ChArUco-family corners are recorded with the pixel convention they were read in.

OpenCV 4.x reports ChArUco corners 0.5 px further on than OpenCV 5 (the
pixel-corner rather than the pixel-centre convention).  A detection cache
written under one must therefore be a miss under the other, and a detection
read from a resized image must be mapped back in its own convention, or a 4.x
corner keeps a bias of ``0.5 * (1 / scale - 1)``.
"""
import numpy as np
import pytest

import pyCamSet.calibration.detection_cache as cache_module
from pyCamSet.calibration_targets import ImageDetection, TargetDetection
from pyCamSet.calibration_targets.core import abstract_target
from pyCamSet.calibration_targets.core.target_registry import build_target
from pyCamSet.calibration_targets.markers import aruco2_gridboard


def _write_cache(tmp_path, spec, offset):
    target = build_target(spec)
    path = tmp_path / "detected_datapoints.npz"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(aruco2_gridboard, "opencv_charuco_pixel_offset", lambda version=None: offset)
        cache_module.save_to_cache(
            TargetDetection(cam_names=["cam0"], data=np.array([[0, 0, 0, 1.0, 2.0]])),
            [(8, 12)], path, target, ["cam0"], None)
    return target, path


def _matches(path, target, offset):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(aruco2_gridboard, "opencv_charuco_pixel_offset", lambda version=None: offset)
        return cache_module.cache_matches(path, target, ["cam0"], None)


@pytest.mark.parametrize("spec", [
    {"type": "ChArUco", "marker_backend": "aruco1"},
    {"type": "ChArUco", "marker_backend": "aruco2"},
    {"type": "Ccube", "marker_backend": "aruco2"},
    {"type": "ChArUco2"},
], ids=["ChArUco-aruco1", "ChArUco-aruco2", "Ccube-aruco2", "ChArUco2"])
def test_a_cache_from_the_other_convention_is_a_miss(tmp_path, spec):
    target, path = _write_cache(tmp_path, spec, 0.5)   # written under OpenCV 4.x
    assert _matches(path, target, 0.0) is False         # read under OpenCV 5
    assert _matches(path, target, 0.5) is True


def test_a_puzzleboard_cache_does_not_depend_on_the_convention(tmp_path):
    target, path = _write_cache(tmp_path, {"type": "PuzzleBoard"}, 0.5)
    assert _matches(path, target, 0.0) is True


def test_convention_names():
    assert aruco2_gridboard.opencv_charuco_pixel_convention("5.0.0") == "pixel-centre"
    assert aruco2_gridboard.opencv_charuco_pixel_convention("4.11.0") == "pixel-corner"


@pytest.mark.parametrize("offset", [0.0, 0.5])
@pytest.mark.parametrize("scale", [0.25, 0.5, 1.5])
def test_the_scaled_restore_maps_in_the_points_own_convention(offset, scale):
    centre = np.array([[10.0, 20.0], [123.25, 7.5]])               # native, pixel-centre
    scaled_centre = (centre + 0.5) * scale - 0.5                     # where the resize puts them
    reported = ImageDetection(np.array([0, 1]), scaled_centre + offset)  # as the detector reports them
    restored = abstract_target._restore_native_coordinates(reported, scale, offset)
    np.testing.assert_allclose(restored.image_points, centre + offset, atol=1e-12)


def test_the_offset_follows_the_target(monkeypatch):
    monkeypatch.setattr(aruco2_gridboard, "opencv_charuco_pixel_offset", lambda version=None: 0.5)
    assert abstract_target._corner_pixel_offset(build_target({"type": "ChArUco"})) == 0.5
    assert abstract_target._corner_pixel_offset(build_target({"type": "ChArUco2"})) == 0.5
    assert abstract_target._corner_pixel_offset(build_target({"type": "PuzzleBoard"})) == 0.0
