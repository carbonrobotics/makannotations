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

import copy
from PyQt5.QtWidgets import QWidget, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout, QListWidget


class SettingsNameWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.settings_window = parent

        layout = QVBoxLayout(self)

        self.edit = QLineEdit()
        layout.addWidget(self.edit)

        button = QPushButton("Save", self)
        button.clicked.connect(self.save)
        layout.addWidget(button)

    def save(self):
        self.settings_window.new_name(self.edit.text())
        self.close()
        self.settings_window.name_window_closed()
        self.deleteLater()

    def closeEvent(self, evt):
        evt.accept()
        self.settings_window.name_window_closed()
        self.deleteLater()


class SettingsWindow(QWidget):
    def __init__(self, parent, settings_get, settings_set, settings_map):
        super().__init__()
        self.main_window = parent

        layout = QVBoxLayout(self)

        button_widget = QWidget(self)
        layout.addWidget(button_widget)
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 0, 0, 0)
        save_button = QPushButton("Save Current", self)
        save_button.clicked.connect(self.save_button)
        button_layout.addWidget(save_button)
        load_button = QPushButton("Load Selected", self)
        load_button.clicked.connect(self.load_button)
        button_layout.addWidget(load_button)

        self.list = QListWidget()
        layout.addWidget(self.list)

        self.name_window = None

        self.settings_get = settings_get
        self.settings_set = settings_set
        self.settings_map = settings_map
        for k in settings_map:
            self.list.addItem(k)

    def new_name(self, name):
        if name not in self.settings_map:
            self.list.addItem(name)
        self.settings_map[name] = copy.deepcopy(self.settings_get())
        self.main_window.save_config()

    def closeEvent(self, evt):
        evt.accept()
        self.main_window.settings_window_closed()
        self.deleteLater()

    def name_window_closed(self):
        self.name_window = None

    def save_button(self, evt):
        if self.name_window is None:
            self.name_window = SettingsNameWindow(self)
            self.name_window.setFixedSize(200, 200)
            self.name_window.show()
        else:
            self.name_window.close()
            self.name_window = None

    def load_button(self, evt):
        item = self.list.currentItem()
        if item is not None and item.text() in self.settings_map:
            settings = self.settings_map[item.text()]
            self.settings_set(settings)
