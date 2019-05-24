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
from typing import Optional
from PyQt5.QtGui import QColor
from abstractions.main_window import MainWindowObserver
from abstractions.filters import FilterObserver, FilterSubject, Filters

color_opacity = 70


class Filter(FilterSubject, MainWindowObserver, abc.ABC):
    """
    Filter stores common logic for ImageFilter and LayerFilter
    """

    NAME: Optional[Filters] = None
    STR_NAME: Optional[str] = None
    COLOR: Optional[QColor] = None

    def __init__(self, observer: FilterObserver):
        self._active: bool = False
        self._observer: FilterObserver = observer

    @property
    def name(self) -> Optional[Filters]:
        return self.NAME

    @property
    def str_name(self) -> Optional[str]:
        return self.STR_NAME

    @property
    def color(self) -> Optional[QColor]:
        return self.COLOR

    @property
    def active(self) -> bool:
        return self._active

    def set_mode(self, active: bool) -> None:
        self._active = active
        """Sends notification about changed mode event to it's observer"""
        self._observer.mode_update(self)

    @abc.abstractmethod
    def skip_image(self, index: int) -> bool:
        pass
