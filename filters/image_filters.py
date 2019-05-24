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
from image_canvas import ImageCanvas
from typing import Dict, List, Callable
from abstractions.filters import Filters, FilterObserver
from abstractions.main_window import MainWindowSubject, MainWindowObserverArgs, MainWindowUpdate
from filters.base import Filter, color_opacity


class ImageFilter(Filter, abc.ABC):
    """
    ImageFilter operates on the whole image (doesn't care about specific layers).
    """

    def __init__(self, observer: FilterObserver):
        super().__init__(observer)
        """self._images: {image_index: value}"""
        self._images: Dict[int, bool] = {}

    @property
    def images(self) -> Dict[int, bool]:
        return self._images

    def _set_images(self, images: List[str], get_image_value: Callable[[str], bool]) -> None:
        self._images.clear()
        for index in range(len(images)):
            self._images[index] = get_image_value(images[index])

        self._observer.images_update(self)

    def _update_image_value(self, index: int, value: bool) -> None:
        self._images[index] = value
        self._observer.image_update(self, index, self._images[index])

    def skip_image(self, index: int) -> bool:
        if self._active and not self._images[index]:
            return True
        return False


class SkipCertifiedImagesFilter(ImageFilter):
    NAME: Filters = Filters.SKIP_CERTIFIED_IMAGES
    STR_NAME: str = "Skip Certified Images"
    COLOR: QColor = QColor(0, 255, 0, color_opacity)

    def set_images(self, subject: MainWindowSubject) -> None:
        def get_image_value(image: str):
            return ImageCanvas.image_has_unncertified_layers(subject.images_path, image, subject.layers)

        self._set_images(subject.images, get_image_value)

    def update_image_value(self, subject: MainWindowSubject, args: MainWindowObserverArgs) -> None:
        if args.update_type in [MainWindowUpdate.MASK, MainWindowUpdate.CERTIFIED]:
            image_value: bool = any(
                [
                    (not certified and has_mask)
                    for certified, has_mask in zip(subject.layers_certified.values(), subject.layers_has_mask.values())
                ]
            )
            self._update_image_value(args.index, image_value)


class HardExampleFilter(ImageFilter):
    NAME: Filters = Filters.HARD_EXAMPLE
    STR_NAME: str = "Hard Example"
    COLOR: QColor = QColor(123, 0, 180, color_opacity)

    def set_images(self, subject: MainWindowSubject) -> None:
        def get_image_value(image: str):
            return ImageCanvas.image_has_hard_example(subject.images_path, image, subject.layers)

        self._set_images(subject.images, get_image_value)

    def update_image_value(self, subject: MainWindowSubject, args: MainWindowObserverArgs) -> None:
        if args.update_type == MainWindowUpdate.HARD_EXAMPLE:
            self._update_image_value(args.index, any(subject.layers_hard_example.values()))
