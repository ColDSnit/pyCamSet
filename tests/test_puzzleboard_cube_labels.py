"""A PuzzleBoardCube face number is readable on every face, at every size."""

from __future__ import annotations

import io
import re

import numpy as np
import pytest

from pyCamSet.calibration_targets.puzzleboard_cube import PuzzleBoardCube


def _label_fill(svg: str) -> str:
    """The fill of the face-number text element in a face texture."""
    match = re.search(r'<text[^>]*fill="(\w+)"', svg)
    assert match, "the face texture carries no numbered label"
    return match.group(1)


def _corner_colour(svg: str) -> str:
    """The colour of the corner cell the label sits in, from the rendered face."""
    cairosvg = pytest.importorskip("cairosvg")
    from PIL import Image

    try:
        png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=400)
    except OSError as exc:  # no native Cairo library on this machine
        pytest.skip(f"cairosvg cannot render here: {exc}")
    image = np.asarray(Image.open(io.BytesIO(png)).convert("L"), dtype=float)
    # Just inside the bottom-left corner, left of where the number is drawn.
    height, width = image.shape
    sample = image[int(height * 0.995), int(width * 0.005)]
    return "black" if sample < 128 else "white"


@pytest.mark.parametrize("n_points", [5, 6])
def test_every_face_label_contrasts_with_its_corner(n_points):
    """An odd square count flips some corner cells to white; the label follows."""
    cube = PuzzleBoardCube(n_points=n_points, length=60.0)
    for face in range(6):
        svg = cube._face_svg(face)
        corner = _corner_colour(svg)
        assert _label_fill(svg) != corner, (
            f"face {face + 1} of a {n_points}-square cube draws its number in {corner} "
            f"on a {corner} corner")
