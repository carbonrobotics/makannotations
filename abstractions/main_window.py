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
from typing import List, Optional, Dict
from aenum import AutoNumberEnum


class MainWindowSubject(abc.ABC):
    """An interface to allow the observers to get properties from MainWindow."""

    @property
    @abc.abstractmethod
    def images(self) -> List[str]:
        pass

    @property
    @abc.abstractmethod
    def images_path(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def layers(self) -> List[str]:
        pass

    @property
    @abc.abstractmethod
    def layers_hard_example(self) -> Dict[str, bool]:
        pass

    @property
    @abc.abstractmethod
    def layers_has_mask(self) -> Dict[str, bool]:
        pass

    @property
    @abc.abstractmethod
    def layers_certified(self) -> Dict[str, bool]:
        pass


class MainWindowUpdate(AutoNumberEnum):
    MASK = ()
    CERTIFIED = ()
    HARD_EXAMPLE = ()


class MainWindowObserverArgs:
    def __init__(self, update_type: MainWindowUpdate, index: int, layer: Optional[str] = None):
        self._update_type: MainWindowUpdate = update_type
        self._index: int = index
        self._layer: Optional[str] = layer

    @property
    def update_type(self) -> MainWindowUpdate:
        return self._update_type

    @property
    def index(self) -> int:
        return self._index

    @property
    def layer(self) -> Optional[str]:
        return self._layer


class MainWindowObserver(abc.ABC):
    """Definition of updates from MainWindow."""

    @abc.abstractmethod
    def set_images(self, main_window: MainWindowSubject) -> None:
        """Observer's handler for updates of MainWindow's images."""
        pass

    @abc.abstractmethod
    def update_image_value(self, main_window: MainWindowSubject, args: MainWindowObserverArgs) -> None:
        """Observer's handler for updates of some MainWindow's image."""
        pass
