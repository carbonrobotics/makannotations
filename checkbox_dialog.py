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

from PyQt5.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout, QCheckBox


class CheckboxDialog(QDialog):
    """
    CheckboxDialog lets the user select items from a list
    """

    def __init__(self, title, items):
        super().__init__()

        self.result = None
        layout = QVBoxLayout(self)
        checks = QVBoxLayout()
        buttons = QHBoxLayout()
        self.checks = []

        cancel = QPushButton(text="Cancel")
        buttons.addWidget(cancel)
        ok = QPushButton(text="Ok")
        buttons.addWidget(ok)

        for item in items:
            qcb = QCheckBox(item)
            checks.addWidget(qcb)
            self.checks.append(qcb)

        cancel.clicked.connect(self.on_cancel)
        ok.clicked.connect(self.on_ok)

        layout.addLayout(checks)
        layout.addLayout(buttons)
        self.setWindowTitle(title)

    def get_checked(self):
        self.exec_()
        if self.result is None or len(self.result) == 0:
            return None
        return self.result

    def on_ok(self):
        self.result = [x.text() for x in self.checks if x.isChecked()]
        self.done(0)

    def on_cancel(self):
        self.done(0)
