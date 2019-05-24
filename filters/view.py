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

from PyQt5.QtWidgets import QCheckBox, QComboBox, QWidget, QHBoxLayout, QLabel, QToolBar, QLayout
from PyQt5.QtGui import QFont
from typing import Dict, List, Optional, Union
from abstractions.filters import Filters
from abstractions.main_window import MainWindowSubject, MainWindowObserver, MainWindowObserverArgs
from filters.base import Filter


"""
Stores classes that are responsible for UI.
"""


class FilterUI(MainWindowObserver):
    """
    Filter representation in UI.
    Contains checkbox and filter logic.
    Each time checkbox change it's state Filter sends notification about changed mode event to it's observer.
    """

    def __init__(self, filter_object: Filter):
        super().__init__()

        """Controls state of the filter (active / not active)"""
        self.checkbox: Optional[QCheckBox] = None

        """Filter logic."""
        self.filter: Filter = filter_object

    @property
    def name(self) -> Filters:
        return self.filter.name

    @property
    def str_name(self) -> str:
        return self.filter.str_name

    @property
    def active(self) -> bool:
        return self.filter.active

    def set_checkbox(self, checkbox: QCheckBox) -> None:
        self.checkbox = checkbox
        """Set filter mode."""
        self.checkbox.stateChanged.connect(lambda: self.filter.set_mode(self.checkbox.isChecked()))

    def skip_image(self, index: int) -> bool:
        return self.filter.skip_image(index)

    def set_images(self, main_window: MainWindowSubject) -> None:
        self.filter.set_images(main_window)

    def update_image_value(self, main_window: MainWindowSubject, args: MainWindowObserverArgs) -> None:
        self.filter.update_image_value(main_window, args)

    def display_filter(self, widget: Union[QToolBar, QLayout]) -> None:
        checkbox = QCheckBox(self.str_name)
        checkbox.setChecked(self.active)
        self.set_checkbox(checkbox)
        widget.addWidget(checkbox)


class LayerFilterUI(FilterUI):
    def __init__(self, filter_object: Filter):
        super().__init__(filter_object)

    def set_active_layer(self, active_layer: str) -> None:
        self.filter.set_active_layer(active_layer)


class LayerFiltersUI(MainWindowObserver):
    """
    Layer filters representation in UI.
    Contains select for setting active_layer and list of LayerFilterUI.
    Each time active_layer is changed LayerFilter's matched images are changed accordingly and
    it sends notification to it's observer about this event."
    """

    def __init__(self, filters: List[LayerFilterUI]):
        super().__init__()

        self._layer_select: Optional[QComboBox] = None

        """Active_layer is the same for all filters."""
        self._active_layer: Optional[str] = None
        self.filters_ui: List[LayerFilterUI] = filters

    def set_layer_select(self, layers_select: QComboBox) -> None:
        self._layer_select = layers_select
        if self._active_layer is not None:
            self._layer_select.setCurrentText(self._active_layer)
        else:
            self._active_layer = layers_select.currentText()
        self._layer_select.currentIndexChanged.connect(lambda: self.set_active_layer(self._layer_select.currentText()))

    def set_active_layer(self, active_layer: str) -> None:
        """
        Each time active_layer is changed LayerFilter sends notification to it's observer about images update event.
        """
        self._active_layer = active_layer
        for filter_ui in self.filters_ui:
            filter_ui.set_active_layer(self._active_layer)

    def set_images(self, main_window: MainWindowSubject) -> None:
        for filter_ui in self.filters_ui:
            filter_ui.set_images(main_window)

    def update_image_value(self, main_window: MainWindowSubject, args: MainWindowObserverArgs) -> None:
        for filter_ui in self.filters_ui:
            filter_ui.update_image_value(main_window, args)

    def skip_image(self, index: int) -> bool:
        return any([filter_ui.skip_image(index) for filter_ui in self.filters_ui])

    def display_filter(self, main_widget: QToolBar) -> None:
        main_widget.addWidget(self._layer_select)

        def add_filters_block(label_name: str, filters_ui: List[FilterUI], toolbar: QToolBar) -> None:
            widget = QWidget()
            layout = QHBoxLayout(widget)
            label = QLabel(label_name)
            label.setFont(QFont("Times", 12))
            layout.addWidget(label)
            for filter_ui in filters_ui:
                filter_ui.display_filter(layout)
            toolbar.addWidget(widget)

        named_filters: Dict[Filters, LayerFilterUI] = LayerFiltersUI.filters_to_dict(self.filters_ui)
        add_filters_block(
            "Has Mask: ", [named_filters[Filters.LAYER_HAS_MASK], named_filters[Filters.LAYER_NO_MASK]], main_widget,
        )
        add_filters_block(
            "Certified:",
            [named_filters[Filters.LAYER_CERTIFIED], named_filters[Filters.LAYER_NOT_CERTIFIED]],
            main_widget,
        )

    @staticmethod
    def filters_to_dict(filters_ui: List[LayerFilterUI]) -> Dict[Filters, LayerFilterUI]:
        return {filter_ui.name: filter_ui for filter_ui in filters_ui}
