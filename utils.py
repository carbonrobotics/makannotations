# Copyright (c) 2020 Maka Autonomous Robotic Systems, Inc.
#
# This file is part of Makannotations.
#
# Makannotations is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Makannotations is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from datetime import datetime
import math
import os
from typing import List, Tuple

import numpy as np


def format_seconds(s: float) -> str:
    """Format seconds as HH:MM:SS"""
    assert s >= 0, "seconds should be positive. Got: %s" % str(s)
    hours = s // 3600
    s = s - (hours * 3600)
    minutes = s // 60
    seconds = s - (minutes * 60)
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))


# You should always use this
# Avoid local time zones whenever possible
def iso8601_timestamp(ms: bool = True) -> str:
    """
    https://en.wikipedia.org/wiki/ISO_8601

    Date                    2019-11-22
    Date and time in UTC    2019-11-22T04:57:59+00:00
                            2019-11-22T04:57:59Z
                            20191122T045759Z
    Week                    2019-W47
    Date with week number   2019-W47-5
    Date without year       --11-22
    Ordinal date            2019-326=
    -------
    :param ms: whether to add millis
    """
    iso_8601_with_millis = datetime.utcnow().isoformat()
    if not ms:
        result = iso_8601_with_millis[:19]
    else:
        result = iso_8601_with_millis
    return result + "Z"


def timestamp_filename(filename: str, ms: bool = True) -> str:
    # Windows doesn't like : in filenames.
    time_fname = iso8601_timestamp(ms=ms).replace(":", "-")
    return "{}.{}".format(filename, time_fname)


def decode_color(color: List[int]) -> Tuple[Tuple[int, int, int], int]:
    """Decodes color into RGB color and texture ID."""
    rgb_color = (color[0], color[1], color[2])
    if len(color) == 4:
        return rgb_color, color[3]
    else:
        return rgb_color, 0


def make_texture(shape: Tuple[int, int], texture_id: int) -> np.ndarray:
    if texture_id == 0:
        base = np.ones((1, 1))

    elif texture_id == 1:
        base = np.array(
            [
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
            ]
        )

    elif texture_id == 2:
        base = np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 1, 1, 1],
                [0, 1, 1, 1, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0, 1, 1, 1, 0],
                [1, 1, 1, 0, 0, 0, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 1, 1],
            ]
        )

    elif texture_id == 3:
        base = np.array(
            [
                [1, 1, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1],
            ]
        )

    elif texture_id == 4:
        base = np.array(
            [
                [1, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 1],
            ]
        )

    else:
        raise ValueError(f"Unknown texture: {texture_id}")

    return np.tile(base.astype(np.bool_), (math.ceil(shape[0] / base.shape[0]), math.ceil(shape[1] / base.shape[1])))[
        : shape[0], : shape[1]
    ]


def get_root_dir() -> str:
    return os.path.dirname(os.path.realpath(__file__))
