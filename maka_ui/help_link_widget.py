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

import os
import webbrowser

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QPushButton

from maka_ui.constants import ICONS_FOLDER
from utils import get_root_dir


class HelpLinkWidget(QPushButton):
    def __init__(self, link: str) -> None:
        super().__init__()
        self.setFixedWidth(15)
        self.setFixedHeight(15)
        self.setStyleSheet(
            """
                QPushButton {border: none;}
            """
        )
        self.setIcon(QIcon(os.path.join(get_root_dir(), ICONS_FOLDER, "question.png")))
        self.setIconSize(QSize(15, 15))
        self.link = link
        self.clicked.connect(self.open)

    def open(self) -> None:
        webbrowser.open(self.link)
