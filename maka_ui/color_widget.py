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

import numpy as np
from PyQt5.QtGui import QPainter, QImage
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QRect

from utils import decode_color, make_texture


class ColorWidget(QLabel):
    WIDTH = 20
    HEIGHT = 15

    def __init__(self, color, *args):
        super().__init__(*args)
        self.setFixedWidth(ColorWidget.WIDTH)
        self.setFixedHeight(ColorWidget.HEIGHT)

        rgb_color, texture_id = decode_color(color)
        mono_texture = make_texture((ColorWidget.HEIGHT, ColorWidget.WIDTH), texture_id)
        texture = (
            np.ones((ColorWidget.HEIGHT, ColorWidget.WIDTH, 3))
            * np.array(rgb_color)
            * mono_texture.reshape(ColorWidget.HEIGHT, ColorWidget.WIDTH, 1)
        )

        self.rect = QRect(0, 0, ColorWidget.WIDTH, ColorWidget.HEIGHT)
        self.image = QImage(texture.astype(np.uint8), ColorWidget.WIDTH, ColorWidget.HEIGHT, QImage.Format_RGB888)
        self.draw_mode = False

    def set_draw_mode(self, mode):
        self.draw_mode = mode
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.draw_mode:
            painter = QPainter()
            painter.begin(self)
            painter.drawImage(self.rect, self.image)
            painter.end()
