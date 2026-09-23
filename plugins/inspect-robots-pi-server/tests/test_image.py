"""Tests for image resize/encode helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PIL import Image

from inspect_robots_pi_server._image import (
    jpeg_encode,
    resample_for,
    resize_stretch,
    resize_with_pad,
    validate_image,
)
from inspect_robots_pi_server._protocol import PiServerError


def test_resize_stretch_shape() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    resized = resize_stretch(image, 5, 5, resample_for("bilinear"))
    assert resized.shape == (5, 5, 3)
    assert resized.dtype == np.uint8


def test_resize_with_pad_shape_and_centering() -> None:
    image = np.full((10, 20, 3), 255, dtype=np.uint8)
    resized = resize_with_pad(image, 20, 20, resample_for("bilinear"), pad_value=0)
    assert resized.shape == (20, 20, 3)
    # A wide source into a square target pads the top/bottom rows with zeros.
    assert np.all(resized[0] == 0)
    assert np.any(resized[10] != 0)


def test_resample_for_rejects_unknown_interpolation() -> None:
    with pytest.raises(PiServerError, match="unknown interpolation"):
        resample_for("nearest")


def test_jpeg_encode_round_trips_approximately() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[4:12, 4:12] = [200, 100, 50]
    encoded = jpeg_encode(image, quality=90)
    decoded = np.asarray(
        Image.open(__import__("io").BytesIO(encoded)).convert("RGB"), dtype=np.uint8
    )
    assert decoded.shape == image.shape
    assert np.mean(np.abs(decoded.astype(int) - image.astype(int))) < 10


@pytest.mark.parametrize(
    "bad, match",
    [
        (np.zeros((4, 4, 3), dtype=np.float32), "must be uint8"),
        (np.zeros((4, 4), dtype=np.uint8), "must be"),
        (np.zeros((4, 4, 4), dtype=np.uint8), "must be"),
        (np.zeros((0, 4, 3), dtype=np.uint8), "zero dimension"),
    ],
)
def test_validate_image_rejects_bad_shapes(bad: np.ndarray[Any, Any], match: str) -> None:
    with pytest.raises(PiServerError, match=match):
        validate_image(bad, "test image")


def test_validate_image_accepts_well_formed_input() -> None:
    validate_image(np.zeros((4, 4, 3), dtype=np.uint8), "test image")
