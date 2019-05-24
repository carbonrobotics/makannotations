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

import abc
from aenum import AutoNumberEnum
from typing import Dict, Optional
from PyQt5.QtGui import QColor


class Filters(AutoNumberEnum):
    SKIP_CERTIFIED_IMAGES = ()
    HARD_EXAMPLE = ()
    LAYER = ()
    LAYER_HAS_MASK = ()
    LAYER_NO_MASK = ()
    LAYER_CERTIFIED = ()
    LAYER_NOT_CERTIFIED = ()


class FilterSubject(abc.ABC):
    """An interface to allow the observer to get properties from filters."""

    @property
    @abc.abstractmethod
    def name(self) -> Optional[Filters]:
        """Filter's name."""
        pass

    @property
    @abc.abstractmethod
    def color(self) -> Optional[QColor]:
        """Color for observer to visualize images data."""
        pass

    @property
    @abc.abstractmethod
    def active(self) -> bool:
        """State of the filter (active / not active)."""
        pass

    @property
    @abc.abstractmethod
    def images(self) -> Dict[int, bool]:
        """Matched images for the observer."""
        pass


class FilterObserver(abc.ABC):
    """Definition of updates from filter."""

    @abc.abstractmethod
    def images_update(self, subject: FilterSubject) -> None:
        """Observer's handler for updates of filter's matched images."""
        pass

    @abc.abstractmethod
    def image_update(self, subject: FilterSubject, index: int, value: bool) -> None:
        """Observer's handler for updates of filter's image value."""
        pass

    @abc.abstractmethod
    def mode_update(self, subject: FilterSubject) -> None:
        """Observer's handler for updates of filter's mode (active / not active)."""
        pass
