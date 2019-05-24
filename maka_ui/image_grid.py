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

from PyQt5.QtWidgets import QButtonGroup, QWidget, QGridLayout, QPushButton, QVBoxLayout, QLabel, QHBoxLayout, QLineEdit
from typing import Optional, List, Callable
from PyQt5.QtCore import Qt
from typing import Callable


class ImageGridSizeSetter(QWidget):
    def __init__(self, set_grid_size: Callable):
        super().__init__()
        self.setFixedSize(200, 200)
        self.setWindowTitle("Set Grid Size:")
        self.main_layout: QVBoxLayout = QVBoxLayout(self)

        self._rows_text_valid: bool = False
        self._columns_text_valid: bool = False

        self.add_rows_widget()
        self.add_columns_widget()
        self.add_set_button(set_grid_size)

    def add_set_button(self, set_grid_size: Callable) -> None:
        def button_click(image_grid_size_setter: ImageGridSizeSetter):
            set_grid_size(
                [int(image_grid_size_setter.rows_input.text()), int(image_grid_size_setter.columns_input.text())]
            )
            self.close()

        self.set_button: QPushButton = QPushButton("Set")
        self.set_button.clicked.connect(lambda: button_click(self))
        self.set_button.setEnabled(False)

        self.main_layout.addWidget(self.set_button)

    def add_widget(self, label: QLabel, input: QLineEdit) -> None:
        widget = QWidget()
        layout = QHBoxLayout(widget)

        layout.addWidget(label)
        layout.addWidget(input)

        self.main_layout.addWidget(widget)

    @staticmethod
    def create_label(name: str) -> QLabel:
        label: QLabel = QLabel(name)
        label.setFixedWidth(80)

        return label

    @staticmethod
    def create_input(text_changed: Callable) -> QLineEdit:
        input: QLineEdit = QLineEdit()
        input.textChanged.connect(text_changed)

        return input

    def add_rows_widget(self) -> None:
        label: QLabel = ImageGridSizeSetter.create_label("Rows:")
        self.rows_input: QLineEdit = ImageGridSizeSetter.create_input(self.rows_text_changed)
        self.add_widget(label, self.rows_input)

    def add_columns_widget(self) -> None:
        label: QLabel = ImageGridSizeSetter.create_label("Columns:")
        self.columns_input: QLineEdit = ImageGridSizeSetter.create_input(self.columns_text_changed)
        self.add_widget(label, self.columns_input)

    @staticmethod
    def text_changed(input: QLineEdit) -> bool:
        try:
            number = int(input.text())
            if number > 0:
                return True
        except ValueError:
            pass

        return False

    def rows_text_changed(self) -> None:
        self._rows_text_valid = ImageGridSizeSetter.text_changed(self.rows_input)
        self.change_set_button_state()

    def columns_text_changed(self) -> None:
        self._columns_text_valid = ImageGridSizeSetter.text_changed(self.columns_input)
        self.change_set_button_state()

    def change_set_button_state(self) -> None:
        self.set_button.setEnabled((self._rows_text_valid and self._columns_text_valid))

    def closeEvent(self, evt):
        evt.accept()
        self.deleteLater()


class ImageGridUI(QWidget):
    def __init__(self, set_rectangle: Callable, rows: int, columns: int):
        super().__init__()
        self._rows: int = rows
        self._columns: int = columns
        self._image_grid: ImageGrid = ImageGrid(set_rectangle, rows, columns)
        self._previous_button_id: Optional[int] = None
        self.create_ui_grid()

    def add_grid_button(self, row: int, column: int) -> QPushButton:
        button: QPushButton = QPushButton(self)
        button.setCheckable(True)
        button.setStyleSheet(
            """
                       QPushButton {background:rgb(250,240,230); border: 1px solid red;}
                       QPushButton::checked{background:rgb(117, 218, 255); border: 1px solid red;}
                   """
        )

        self.grid_layout.addWidget(button, row, column)
        button.setFixedHeight(20)
        button.setFixedWidth(20)

        return button

    def create_ui_grid(self) -> None:
        self.grid_buttons: QButtonGroup = QButtonGroup()
        self.grid_buttons.setExclusive(True)
        self.main_layout: QVBoxLayout = QVBoxLayout(self)
        self.main_layout.setSpacing(1)
        self.main_layout.setContentsMargins(0, 0, 5, 0)
        label: QLabel = QLabel("Crop Image")
        label.setAlignment(Qt.AlignCenter)
        label.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(label)
        widget: QWidget = QWidget()
        self.grid_layout: QGridLayout = QGridLayout(widget)
        self.main_layout.addWidget(widget)
        self.grid_layout.setSpacing(0)
        self.grid_layout.setSizeConstraint(3)
        for r in range(self._rows):
            for c in range(self._columns):
                self.grid_buttons.addButton(self.add_grid_button(r, c), self._columns * r + c)

        self.grid_buttons.buttonClicked[int].connect(self.set_rectangle)

    def set_rectangle(self, id: int) -> None:
        button: QPushButton = self.grid_buttons.button(id)
        if button.isChecked() and id == self._previous_button_id:
            self.grid_buttons.setExclusive(False)
            button.setChecked(False)
            self._previous_button_id = None
            self.grid_buttons.setExclusive(True)
        else:
            self._previous_button_id = id
        self._image_grid.set_rectangle(button.isChecked(), id)

    def _move_horizontally(self, direction: int) -> None:
        if self._previous_button_id is None:
            return
        id: int = self._previous_button_id + direction
        if 0 <= id < self._rows * self._columns:
            button: QPushButton = self.grid_buttons.button(id)
            button.animateClick()

    def go_left(self) -> None:
        self._move_horizontally(-1)

    def go_right(self) -> None:
        self._move_horizontally(1)


class ImageGrid:
    MARGIN_PERCENT = 0.2

    def __init__(self, set_rectangle: Callable, rows: int, columns: int):
        self._set_rectangle: Callable = set_rectangle
        self._rows: int = rows
        self._columns: int = columns
        self._cell_width: float = 1 / columns
        self._cell_width_margin: float = self._cell_width * ImageGrid.MARGIN_PERCENT

        self._cell_height: float = 1 / rows
        self._cell_height_margin: float = self._cell_height * ImageGrid.MARGIN_PERCENT

    def normalized_rectangle(self, id: int) -> List[float]:
        row: int = id // self._columns
        column: int = id % self._columns
        start_width = max(float(0), self._cell_width * column - self._cell_width_margin)
        start_height = max(float(0), self._cell_height * row - self._cell_height_margin)
        end_width = min(float(1), self._cell_width * (column + 1) + self._cell_width_margin)
        end_height = min(float(1), self._cell_height * (row + 1) + self._cell_height_margin)
        return [start_width, start_height, end_width, end_height]

    def set_rectangle(self, set_limits: bool, id: int) -> None:
        if set_limits:
            self._set_rectangle(self.normalized_rectangle(id))
        else:
            # no limits
            self._set_rectangle([0, 0, 1, 1])
