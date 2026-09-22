"""Translate the three rosboard message shapes this adapter uses, to and from NumPy.

rosboard performs all ROS message conversion server-side and hands the client
plain JSON: nested ROS submessages arrive as same-named nested dicts,
recursively (confirmed against rosboard's own server-side ``ros2dict()``), so
no ROS message classes are needed client-side.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from io import BytesIO
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, UnidentifiedImageError


def parse_compressed_image(
    payload: Mapping[str, Any], *, resize: tuple[int, int] | None = None
) -> npt.NDArray[np.uint8]:
    """Decode rosboard's ``_data_jpeg`` base64 field into a copied ``(H, W, 3)`` RGB uint8 array.

    Identical decode path to ``inspect_robots_ros._msgs.parse_compressed_image``
    (base64 -> Pillow -> RGB -> contiguous uint8 copy), keyed on rosboard's own
    field name instead of rosbridge's ``data``: rosboard always compresses
    image topics server-side to this field regardless of the source ROS
    message type (plain ``Image`` or ``CompressedImage``).

    ``resize``, when given, is a target ``(height, width)`` applied client-side
    with Pillow after decode: rosboard's own server-side resize is a fixed
    800px-max-dimension stride downsample (``compression.py``), not a
    configurable per-topic target, so matching a declared camera shape other
    than the server's own choice has to happen here.
    """
    data = payload.get("_data_jpeg")
    if not isinstance(data, str):
        raise ValueError("rosboard image payload missing base64 string field '_data_jpeg'")
    try:
        encoded = base64.b64decode(data, validate=True)
        with Image.open(BytesIO(encoded)) as image:
            rgb = image.convert("RGB")
            if resize is not None:
                height, width = resize
                rgb = rgb.resize((width, height))
            return np.asarray(rgb, dtype=np.uint8).copy()
    except (binascii.Error, UnidentifiedImageError, OSError) as exc:
        image_format = payload.get("format", "unknown")
        raise ValueError(
            f"could not decode rosboard image format {image_format!r} as JPEG: {exc}"
        ) from exc


def parse_odometry(
    payload: Mapping[str, Any],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return ``(pose[7], twist[6])`` from a ``nav_msgs/msg/Odometry`` payload.

    ``pose`` is ``[x, y, z, qw, qx, qy, qz]``: position plus orientation
    reordered from ROS's native xyzw to wxyz, the same documented convention
    ``inspect_robots_ros._msgs.parse_pose_stamped`` uses, kept for consistency
    across the framework. ``twist`` is
    ``[linear.x, linear.y, linear.z, angular.x, angular.y, angular.z]`` in
    native units (m/s, rad/s); no rotation representation is involved so no
    reordering applies.
    """
    try:
        pose = payload["pose"]["pose"]
        position, orientation = pose["position"], pose["orientation"]
        pose_out = np.asarray(
            (
                position["x"],
                position["y"],
                position["z"],
                orientation["w"],
                orientation["x"],
                orientation["y"],
                orientation["z"],
            ),
            dtype=np.float64,
        )
        twist = payload["twist"]["twist"]
        linear, angular = twist["linear"], twist["angular"]
        twist_out = np.asarray(
            (linear["x"], linear["y"], linear["z"], angular["x"], angular["y"], angular["z"]),
            dtype=np.float64,
        )
        return pose_out, twist_out
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Odometry must contain numeric pose.pose.{position,orientation} and "
            "twist.twist.{linear,angular} fields"
        ) from exc


def parse_imu(payload: Mapping[str, Any]) -> npt.NDArray[np.float64]:
    """Return ``[qw,qx,qy,qz, wx,wy,wz, ax,ay,az]`` (shape ``(10,)``) from an ``Imu`` payload.

    Orientation is reordered to wxyz (same convention as :func:`parse_odometry`);
    ``angular_velocity`` (rad/s) and ``linear_acceleration`` (m/s^2) pass
    through unchanged since neither carries a rotation representation.
    """
    try:
        orientation = payload["orientation"]
        angular_velocity = payload["angular_velocity"]
        linear_acceleration = payload["linear_acceleration"]
        return np.asarray(
            (
                orientation["w"],
                orientation["x"],
                orientation["y"],
                orientation["z"],
                angular_velocity["x"],
                angular_velocity["y"],
                angular_velocity["z"],
                linear_acceleration["x"],
                linear_acceleration["y"],
                linear_acceleration["z"],
            ),
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Imu must contain numeric orientation, angular_velocity, and linear_acceleration fields"
        ) from exc


def build_twist(linear_x: float, angular_z: float) -> dict[str, Any]:
    """Build a ``geometry_msgs/msg/Twist`` fields dict for the drive command.

    Only ``linear.x`` and ``angular.z`` are commanded (ground robot); the
    remaining four degrees of freedom are wire-required zeros.
    """
    return {
        "linear": {"x": float(linear_x), "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": float(angular_z)},
    }
