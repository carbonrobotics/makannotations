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

from PyQt5.QtWidgets import QDialog, QHBoxLayout, QListWidget, QPushButton, QVBoxLayout

import boto3  # noqa
from s3utils import get_directory_listing


class S3ListDialog(QDialog):
    """
    S3ListDialog is file open dialog style s3 directory selector.
    """

    def __init__(self, prefix):
        super().__init__()

        self.result = None
        self.prefix = prefix
        self.dir_stack = self.prefix.split("/")
        layout = QVBoxLayout(self)
        buttons = QHBoxLayout()

        list = QListWidget()
        list.doubleClicked.connect(self.on_double_clicked)
        self.populate(list)
        self.list = list

        cancel = QPushButton(text="Cancel")
        buttons.addWidget(cancel)
        open = QPushButton(text="Open")
        buttons.addWidget(open)

        cancel.clicked.connect(self.on_cancel)
        open.clicked.connect(self.on_open)

        layout.addWidget(list)
        layout.addLayout(buttons)

    def populate(self, list):
        print(self.dir_stack)
        path = "/".join(self.dir_stack)
        dirs = get_directory_listing(path)

        if len(self.dir_stack) > 1:
            list.addItem("..")

        for item in dirs:
            list.addItem(item)

    def getSelection(self):
        self.exec_()
        if self.result is None:
            return None
        return self.result

    def on_open(self):
        item = self.list.currentItem()
        if item.text() == "..":
            return
        self.dir_stack.append(item.text())
        self.result = "/".join(self.dir_stack)
        self.done(0)

    def on_cancel(self):
        self.done(0)

    def on_double_clicked(self):
        item = self.list.currentItem()
        if item.text() == "..":
            if len(self.dir_stack) > 1:
                self.dir_stack = self.dir_stack[:-1]
        else:
            self.dir_stack.append(item.text())
        self.list.clear()
        self.populate(self.list)
