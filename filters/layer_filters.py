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

from PyQt5.QtGui import QColor
import abc
import functools
from image_canvas import ImageCanvas
from typing import Dict, List, Callable, Union, Optional
from abstractions.main_window import MainWindowSubject, MainWindowUpdate, MainWindowObserverArgs
from abstractions.filters import FilterObserver, Filters
from filters.base import Filter, color_opacity


class LayerFilter(Filter, abc.ABC):
    """
    LayerFilter cares about layer on which it operates.
    """

    def __init__(self, observer: FilterObserver):
        super().__init__(observer)
        """"
        self._images: {layer: {image_index: value}}
        Images data is stored for each layer.
        """
        self._images: Dict[str, Dict[int, bool]] = {}
        self._active_layer: Optional[str] = None

    def set_active_layer(self, active_layer: str) -> None:
        """Observer gets notification about matched images update event every time active_layer is changed."""
        self._active_layer = active_layer
        if self._images.get(self._active_layer):
            self._observer.images_update(self)

    @property
    def active_layer(self) -> Optional[str]:
        """Active layer."""
        return self._active_layer

    @property
    def images(self) -> Dict[int, bool]:
        """Images of active layer."""
        return self._images[self._active_layer]

    def _set_images(
        self,
        images: List[str],
        layers: List[str],
        get_image_value: Union[Callable[[str, str], bool], functools.partial],
    ) -> None:
        self._images.clear()
        for layer in layers:
            self._images[layer]: Dict[int, bool] = {}
            for index in range(len(images)):
                self._images[layer][index] = get_image_value(images[index], layer)

        self._observer.images_update(self)

    def _update_image_value(self, layer: str, index: int, value: bool) -> None:
        self._images[layer][index] = value
        if layer == self._active_layer:
            self._observer.image_update(self, index, self._images[self._active_layer][index])

    def skip_image(self, index: int) -> bool:
        if self._active and not self._images[self._active_layer][index]:
            return True
        return False


class LayerHasMask(LayerFilter):
    NAME: Filters = Filters.LAYER_HAS_MASK
    STR_NAME: str = "Yes"
    COLOR: QColor = QColor(255, 0, 0, color_opacity)

    def set_images(self, subject: MainWindowSubject) -> None:
        self._set_images(
            subject.images, subject.layers, functools.partial(ImageCanvas.layer_has_mask, subject.images_path)
        )

    def update_image_value(self, subject: MainWindowSubject, args: MainWindowObserverArgs) -> None:
        if args.update_type == MainWindowUpdate.MASK:
            self._update_image_value(args.layer, args.index, subject.layers_has_mask[args.layer])


class LayerHasNoMask(LayerFilter):
    NAME: Filters = Filters.LAYER_NO_MASK
    STR_NAME: str = "No"
    COLOR: QColor = QColor(255, 255, 0, color_opacity)

    def set_images(self, subject: MainWindowSubject) -> None:
        self._set_images(
            subject.images,
            subject.layers,
            lambda image, layer: not ImageCanvas.layer_has_mask(subject.images_path, image, layer),
        )

    def update_image_value(self, subject: MainWindowSubject, args: MainWindowObserverArgs) -> None:
        if args.update_type == MainWindowUpdate.MASK:
            self._update_image_value(args.layer, args.index, not subject.layers_has_mask[args.layer])


class LayerCertified(LayerFilter):
    NAME: Filters = Filters.LAYER_CERTIFIED
    STR_NAME: str = "Yes"
    COLOR = QColor(0, 0, 255, color_opacity)

    def set_images(self, subject: MainWindowSubject):
        self._set_images(
            subject.images, subject.layers, functools.partial(ImageCanvas.layer_certified, subject.images_path)
        )

    def update_image_value(self, subject: MainWindowSubject, args: MainWindowObserverArgs):
        if args.update_type == MainWindowUpdate.CERTIFIED:
            self._update_image_value(args.layer, args.index, subject.layers_certified[args.layer])


class LayerNotCertified(LayerFilter):
    NAME: Filters = Filters.LAYER_NOT_CERTIFIED
    STR_NAME: str = "No"
    COLOR: QColor = QColor(0, 255, 255, color_opacity)

    def set_images(self, subject: MainWindowSubject) -> None:
        self._set_images(
            subject.images,
            subject.layers,
            lambda image, layer: not ImageCanvas.layer_certified(subject.images_path, image, layer),
        )

    def update_image_value(self, subject: MainWindowSubject, args: MainWindowObserverArgs) -> None:
        if args.update_type == MainWindowUpdate.CERTIFIED:
            self._update_image_value(args.layer, args.index, not subject.layers_certified[args.layer])
