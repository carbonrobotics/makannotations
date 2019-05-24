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

from PyQt5.QtWidgets import QWidget, QPushButton, QLineEdit, QVBoxLayout, QShortcut
from PyQt5.QtCore import Qt


class SearchWindow(QWidget):
    def __init__(self, go_to_image, search_image):
        super().__init__()
        self.setWindowModality(Qt.ApplicationModal)
        self.setFixedSize(400, 200)
        self.setWindowTitle("Go to image")

        layout = QVBoxLayout(self)

        def text_changed():
            enabled = search_image(self.search_input.text())
            self.go_button.setEnabled(enabled)
            self.go_shortcut.setEnabled(enabled)

        self.search_input = QLineEdit()
        self.search_input.textChanged.connect(text_changed)
        layout.addWidget(self.search_input)

        def cb():
            go_to_image(self.search_input.text())
            self.close()

        self.go_to_image = cb
        self.go_button = QPushButton("Go!")
        self.go_button.clicked.connect(self.go_to_image)
        self.go_button.setEnabled(False)

        self.go_shortcut = QShortcut(Qt.Key_Return, self)
        self.go_shortcut.activated.connect(self.go_to_image)
        self.go_shortcut.setEnabled(False)

        layout.addWidget(self.go_button)

    def closeEvent(self, evt):
        evt.accept()
        self.deleteLater()
