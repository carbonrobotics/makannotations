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

from PyQt5.QtWidgets import QSlider
from PyQt5.QtGui import QPainter, QPen, QBrush
from PyQt5.QtCore import Qt
from abstractions.filters import FilterObserver, FilterSubject


class NavigationSliderMeta(type(QSlider), type(FilterObserver)):
    pass


class NavigationSlider(QSlider, FilterObserver, metaclass=NavigationSliderMeta):
    def __init__(self, *args):
        super().__init__(*args)
        self.handle_width = 9
        self.images = {}
        self.draw_mode = {}
        self.colors = {}

    def images_update(self, subject: FilterSubject):
        self.images[subject.name] = subject.images.copy()
        self.update()

    def image_update(self, subject: FilterSubject, index: int, value: bool):
        if self.images[subject.name][index] != value:
            self.images[subject.name][index] = value
            self.update()

    def mode_update(self, subject: FilterSubject):
        self.draw_mode[subject.name] = subject.active
        self.colors[subject.name] = subject.color
        self.update()

    def paint_over_unavailable_regions(self, filter_name):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.transparent)
        painter.setPen(pen)
        brush = QBrush(self.colors[filter_name], Qt.SolidPattern)
        painter.setBrush(brush)
        if len(self.images.get(filter_name, [])) == 0:
            return
        width = painter.device().width() - 2 * self.handle_width
        height = painter.device().height()
        step = width / len(self.images[filter_name])
        position = self.handle_width
        index = 0
        while index < len(self.images[filter_name]):
            start = position
            while index < len(self.images[filter_name]) and not self.images[filter_name][index]:
                position += step
                index += 1

            if start != position:
                painter.drawRect(start, 0, position - start, height)

            position += step
            index += 1
        painter.end()

    def paintEvent(self, e):
        super().paintEvent(e)
        active_filters = {filter_name: value for filter_name, value in self.draw_mode.items() if value}
        for filter_name in active_filters.keys():
            self.paint_over_unavailable_regions(filter_name)
