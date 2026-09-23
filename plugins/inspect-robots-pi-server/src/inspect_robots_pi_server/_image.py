"""Resize and JPEG-encode an observation image for the PI protocol.

Ported from ``pi_inference_client.preprocessing`` (``resize_image``/
``resize_image_with_pad``/``jpeg_encode``), using Pillow only (no optional
``simplejpeg`` fast path): matches the ``pillow>=10`` convention already used
by ``inspect-robots-ros``/``inspect-robots-rosboard``, and upstream itself
falls back to identical PIL code when ``simplejpeg`` isn't installed, so this
stays bit-for-bit compatible with that fallback path.

``Observation.images[key]`` is already ``(H, W, 3)`` uint8 RGB — exactly this
protocol's wire shape before encoding — so there is no channel permute or
float normalization step here, unlike the in-process LeRobot policy plugin.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import numpy.typing as npt
from PIL import Image

from inspect_robots_pi_server._protocol import PiServerError

_RESAMPLE = {"bilinear": Image.BILINEAR, "lanczos": Image.LANCZOS}


def validate_image(image: npt.NDArray[np.uint8], label: str) -> None:
    """Confirm ``image`` is a well-formed ``(H, W, 3)`` uint8 array."""
    if not isinstance(image, np.ndarray):
        raise PiServerError(
            "invalid_frame", f"{label} is not a NumPy array: {type(image).__name__}"
        )
    if image.dtype != np.uint8:
        raise PiServerError("invalid_frame", f"{label} must be uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise PiServerError("invalid_frame", f"{label} must be (H, W, 3), got shape {image.shape}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise PiServerError("invalid_frame", f"{label} has a zero dimension: {image.shape}")


def resample_for(interpolation: str) -> int:
    """The Pillow resample constant for a protocol ``interpolation`` name."""
    try:
        return int(_RESAMPLE[interpolation])
    except KeyError as exc:
        raise PiServerError("invalid_frame", f"unknown interpolation {interpolation!r}") from exc


def resize_stretch(
    image: npt.NDArray[np.uint8], height: int, width: int, resample: int
) -> npt.NDArray[np.uint8]:
    """Resize ``image`` to exactly ``(height, width)``, distorting aspect ratio."""
    with Image.fromarray(image, mode="RGB") as pil_image:
        resized = pil_image.resize((width, height), resample=resample)
        return np.asarray(resized, dtype=np.uint8)


def resize_with_pad(
    image: npt.NDArray[np.uint8], height: int, width: int, resample: int, pad_value: int = 0
) -> npt.NDArray[np.uint8]:
    """Resize ``image`` to fit within ``(height, width)``, preserving aspect ratio via padding."""
    source_height, source_width = image.shape[:2]
    scale = min(height / source_height, width / source_width)
    scaled_height = max(1, round(source_height * scale))
    scaled_width = max(1, round(source_width * scale))
    with Image.fromarray(image, mode="RGB") as pil_image:
        scaled = pil_image.resize((scaled_width, scaled_height), resample=resample)
        scaled_array = np.asarray(scaled, dtype=np.uint8)

    canvas = np.full((height, width, 3), pad_value, dtype=np.uint8)
    top = (height - scaled_height) // 2
    left = (width - scaled_width) // 2
    canvas[top : top + scaled_height, left : left + scaled_width] = scaled_array
    return canvas


def jpeg_encode(image: npt.NDArray[np.uint8], quality: int = 85) -> bytes:
    """JPEG-encode an ``(H, W, 3)`` uint8 RGB array."""
    buffer = BytesIO()
    with Image.fromarray(image, mode="RGB") as pil_image:
        pil_image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()
