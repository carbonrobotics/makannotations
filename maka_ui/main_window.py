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

import base64
from collections import OrderedDict
import functools
import glob
import json
import os
import pathlib
import ssl
import sys
import urllib.request
import zlib
import logging
import logging.handlers
import numpy as np
from aenum import AutoNumberEnum
from typing import Dict, Union, List, Any, Optional
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QColor, QFont, QIcon, QKeySequence, QPalette
from PyQt5.QtWidgets import (
    QAction,
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QShortcut,
    QSlider,
    QSpinBox,
    QSizePolicy,
    QCheckBox,
    QComboBox,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

import torchscript_model
from image_canvas import ImageCanvas, Settings
from mask_image import load_image
from metadata import DirectoryMetadata
from s3list_dialog import S3ListDialog
from checkbox_dialog import CheckboxDialog
from s3utils import get_file_listing
from utils import timestamp_filename
from maka_ui.navigation import NavigationSlider
from maka_ui.color_widget import ColorWidget
from maka_ui.help_link_widget import HelpLinkWidget
from maka_ui.search_window import SearchWindow
from maka_ui.settings_windows import SettingsWindow
from filters.image_filters import SkipCertifiedImagesFilter, HardExampleFilter
from filters.layer_filters import LayerHasMask, LayerHasNoMask, LayerCertified, LayerNotCertified
from filters.view import FilterUI, LayerFiltersUI, LayerFilterUI
from abstractions.main_window import MainWindowSubject, MainWindowObserverArgs, MainWindowUpdate
from abstractions.filters import Filters
from maka_ui.constants import *
from maka_ui.image_grid import ImageGridUI, ImageGridSizeSetter
from utils import get_root_dir


default_font = QFont("Arial", 10)


class Algorithms(AutoNumberEnum):
    DL_AUTOMASK = ()
    LAB_AUTOMASK = ()
    GRABCUT = ()
    CLUSTERING = ()
    MASK_MANIPULATION = ()
    MOVING_MASKS = ()
    FILTERS = ()


ALGORITHMS_TO_STRING = {
    Algorithms.DL_AUTOMASK: "DL automask",
    Algorithms.LAB_AUTOMASK: "LAB automask",
    Algorithms.GRABCUT: "GrabCut",
    Algorithms.CLUSTERING: "Clustering",
    Algorithms.MASK_MANIPULATION: "Mask manipulation",
    Algorithms.MOVING_MASKS: "Moving Masks",
    Algorithms.FILTERS: "Filters",
}


class LayersStatus(AutoNumberEnum):
    CERTIFICATION = ()
    HARD_EXAMPLE = ()


class MainWindowMeta(type(QMainWindow), type(MainWindowSubject)):
    pass


class MainWindow(QMainWindow, MainWindowSubject, metaclass=MainWindowMeta):
    @staticmethod
    def align_widget_center(current_widget):
        wrap_widget = QWidget()
        qlayout = QHBoxLayout(wrap_widget)
        qlayout.addWidget(current_widget)
        qlayout.setAlignment(Qt.AlignCenter)

        return wrap_widget

    def disable_hotkeys(self):
        for hk in self.hotkeys_list:
            hk.setEnabled(False)

    def enable_hotkeys(self):
        for hk in self.hotkeys_list:
            hk.setEnabled(True)

    def add_toolbar_label(self, text, font_size=20, toolbar=None):
        if toolbar is None:
            toolbar = self.main_toolbar
        label = QLabel(text)
        label.setFont(QFont("Times", font_size, QFont.Bold))
        toolbar.addWidget(label)
        toolbar.addSeparator()

    def update_layers_moving_masks_mode(self):
        for button in self.destination_layers_grp.buttons():
            button.setEnabled(not self.moving_mask_mode)
        self.destination_layers_grp.button(self.layers_grp.checkedId()).setEnabled(False)

        self.main_toolbar.setEnabled(not self.moving_mask_mode)

    def toggle_moving_masks(self):
        destination_layer_button = self.destination_layers_grp.checkedButton()
        cur_layer = self.layers_grp.checkedButton()
        if destination_layer_button is None or destination_layer_button.text() == cur_layer.text():
            self.moving_mask_mode = False
            self.moving_masks_button.setChecked(False)
            destination_layer = None
        elif destination_layer_button.text() != cur_layer.text():
            self.moving_mask_mode = not self.moving_mask_mode
            destination_layer = destination_layer_button.text()

        self.ic.set_moving_masks(destination_layer, self.moving_mask_mode)
        self.update_drawing_status()
        self.update_layers_moving_masks_mode()

    # moving masks to another layer
    def add_moving_masks(self):
        # Layers
        self.algorithm_toolbar.addWidget(QLabel("Destination Layers"))
        self.destination_layers_grp = QButtonGroup(self)
        for count, layer in enumerate(self._layers):
            self.add_toolbar_radio(layer, group=self.destination_layers_grp, widget=self.algorithm_toolbar)
        self.destination_layers_grp.button(self.layers_grp.checkedId()).setEnabled(False)

        self.moving_masks_button = self.add_toolbar_mode_button(
            "MOVE M[A]SKS", self.toggle_moving_masks, self.algorithm_toolbar, tooltip="A", hotkey=ord("A"),
        )

    def add_lab_automask(self):
        self.add_toolbar_label(" Pop green", font_size=16, toolbar=self.algorithm_toolbar)
        self.add_button("LAB automask", self.ic.lab_auto_mask, bar=self.algorithm_toolbar)

    def add_dl_automask(self):
        self.add_toolbar_label("Deep Learning Mask", font_size=16, toolbar=self.algorithm_toolbar)
        self.add_button("Load TorchScript DL model", self.load_torchscript_model, bar=self.algorithm_toolbar)

        self.dl_model_channel = QSpinBox()
        self.dl_model_channel.setPrefix("channel ")
        self.algorithm_toolbar.addWidget(self.dl_model_channel)

        self.add_button("Generate Auto Mask (DL)", self.auto_mask_torchscript, bar=self.algorithm_toolbar)

    def add_clustering(self):
        # Clustering
        self.add_toolbar_label(" Clustering", font_size=16, toolbar=self.algorithm_toolbar)
        self.add_button(
            "1) Rectangle",
            self.ic.toggle_box_drawer,
            hotkey=ord("R"),
            bar=self.algorithm_toolbar,
            tooltip="Optional. To perform operation inside the box",
        )
        self.add_button("2) Seed", self.ic.choose_seed, bar=self.algorithm_toolbar, tooltip="Choose color to pop")

        sliders = {}
        settings = self.ic.get_settings()
        sliders[Settings.Sliders.CLUSTERING] = self.add_toolbar_slider(
            self.ic.set_clusters_amount,
            settings.clusters_amount,
            "Clusters amount",
            min_range=1,
            max_range=100,
            toolbar=self.algorithm_toolbar,
            tooltip="Choose amount of clusters for clustering algorithm.",
        )
        self.ic.set_sliders(sliders)

        self.add_button("4) Clustering", self.ic.clustering, bar=self.algorithm_toolbar, tooltip="Run clustering")

    def add_grabcut(self):
        # GrabCut
        self.add_toolbar_label(" GrabCut", font_size=16, toolbar=self.algorithm_toolbar)
        self.add_button("1) Rectangle", self.ic.toggle_box_drawer, bar=self.algorithm_toolbar, tooltip="Draw the box")
        self.add_button(
            "2) GrabCut",
            self.ic.grab_cut_with_rectangle,
            bar=self.algorithm_toolbar,
            tooltip="Run grabcut algorithm inside this box",
        )

    def add_mask_manipulation(self):
        self.add_toolbar_label("Mask manipulations", toolbar=self.algorithm_toolbar)

        sliders = {}
        settings = self.ic.get_settings()

        sliders[Settings.Sliders.REMOVING_OBJECTS] = self.add_toolbar_slider(
            self.ic.set_removing_objects_size,
            settings.removing_objects_size,
            "Size of objects to remove",
            max_range=1000,
            toolbar=self.algorithm_toolbar,
            tooltip="Will remove small objects from the mask",
        )

        sliders[Settings.Sliders.CLOSING] = self.add_toolbar_slider(
            self.ic.set_closing_iterations_image,
            settings.closing_iterations_image,
            "Closing iterations for clustering",
            max_range=20,
            toolbar=self.algorithm_toolbar,
            tooltip="Will close holes on the mask",
        )

        sliders[Settings.Sliders.DILATION] = self.add_toolbar_slider(
            self.ic.set_dilation_iterations,
            settings.dilation_iterations,
            "Dilation iterations",
            max_range=10,
            toolbar=self.algorithm_toolbar,
            tooltip="Will expand mask",
        )

        sliders[Settings.Sliders.EROSION] = self.add_toolbar_slider(
            self.ic.set_erosion_iterations,
            settings.erosion_iterations,
            "Erosion iterations",
            max_range=10,
            toolbar=self.algorithm_toolbar,
            tooltip="Will reduce mask",
        )

        self.ic.set_sliders(sliders)

    def add_filters(self):
        self.add_toolbar_label("Filters", toolbar=self.algorithm_toolbar)

        self.algorithm_toolbar.addWidget(QLabel("Image filters"))
        self.filters[Filters.SKIP_CERTIFIED_IMAGES].display_filter(self.algorithm_toolbar)
        self.filters[Filters.HARD_EXAMPLE].display_filter(self.algorithm_toolbar)
        self.algorithm_toolbar.addSeparator()

        self.algorithm_toolbar.addWidget(QLabel("Layer filters"))
        layer_select = QComboBox()
        layer_select.addItems(self._layers)
        self.filters[Filters.LAYER].set_layer_select(layer_select)
        self.filters[Filters.LAYER].display_filter(self.algorithm_toolbar)

    def algorithms_stack(self):
        # For displaying a stack of applied algorithms
        self.algorithms_stack_text = QLabel()
        self.algorithms_stack_text.setFixedWidth(300)
        self.algorithms_stack_text.setWordWrap(True)
        self.algorithm_toolbar.addWidget(self.algorithms_stack_text)
        self.ic.set_algorithms_stack_widget(self.algorithms_stack_text)
        self.algorithms_stack_text.show()

    def add_algorithms_button(self, text, function):
        action = QAction(text, self)

        def cb():
            self.algorithm_toolbar.clear()
            self.destination_layers_grp = None

            function()

            self.algorithms_stack()
            self.algorithm_toolbar.show()
            self.hide_button.show()

        action.triggered.connect(cb)
        self.algorithms_menu.addAction(action)

    def add_button(self, text, cb, hotkey=None, bar=None, tooltip=None):
        if bar is None:
            bar = self.main_toolbar
        action = QAction(text, self)
        bar.addAction(action)
        action.triggered.connect(cb)
        if tooltip is not None:
            action.setToolTip(tooltip)
        bar.addSeparator()

        if hotkey is not None:
            action.setShortcut(hotkey)

    @staticmethod
    def add_settings_slider(cb, val, main_layout, settings=None, icon=None, min_range=0, max_range=64, tooltip=None):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        if icon is not None:
            label = QLabel()
            icon = QIcon(os.path.join(get_root_dir(), ICONS_FOLDER, icon))
            pixmap = icon.pixmap(QSize(25, 25))
            label.setPixmap(pixmap)
            layout.addWidget(label)

        slider = QSlider(Qt.Horizontal)
        slider.setSingleStep(1)
        slider.setPageStep(1)  # Windows emits page steps for single mousewheel clicks.
        slider.setRange(min_range, max_range)
        slider.setValue(val)
        slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        value_label = QLabel()
        value_label.setFont(default_font)
        value_label.setText("%.2d" % val)
        value_label.setFixedWidth(30)

        layout.addWidget(slider)
        layout.addWidget(value_label)

        def pass_slide():
            value_label.setText("%.2d" % slider.sliderPosition())
            cb(slider.sliderPosition())

        slider.valueChanged.connect(pass_slide)

        if settings is not None:
            settings.append(slider)
        widget.setContentsMargins(0, 0, 0, 0)
        widget.setMaximumWidth(400)
        if tooltip is not None:
            widget.setToolTip(tooltip)
        main_layout.addWidget(widget)

    def add_drawing_button(self, cb, widget=None, mode=None, tooltip=None, icon=None, hotkey=None):
        button = QPushButton(self)
        button.setCheckable(True)
        button.setStyleSheet(
            """
                QPushButton {background:rgb(250,240,230); border: none;}
                QPushButton::checked{background:rgb(117, 218, 255); border: none;}
            """
        )
        if widget is None:
            widget = self.drawing_toolbar

        widget.addWidget(button)
        if cb is not None:
            button.clicked.connect(cb)
        if tooltip is not None:
            button.setToolTip(tooltip)

        button.setFixedHeight(50)
        button.setFixedWidth(50)
        if icon is not None:
            button.setIcon(QIcon(os.path.join(get_root_dir(), ICONS_FOLDER, icon)))
            button.setIconSize(QSize(40, 40))

        if hotkey is not None:
            button.setShortcut(hotkey)

        if mode is not None:
            self.mode_to_btn[mode] = button

        return button

    def add_toolbar_mode_button(
        self, text, action, widget, mode=None, tooltip=None, hotkey=None, fixed_height=None, fixed_width=None
    ):
        button = QPushButton(text)

        button.setFont(default_font)
        button.setText(text)
        palette = QPalette(button.palette())
        palette.setColor(QPalette.ButtonText, QColor("black"))
        button.setPalette(palette)

        button.setContentsMargins(20, 20, 20, 20)

        button.setCheckable(True)
        button.setStyleSheet(
            """
                       QPushButton {background:rgb(250,240,230); border: none;}
                       QPushButton::checked{background:rgb(117, 218, 255); border: none;}
                   """
        )

        align_widget = self.align_widget_center(button)
        widget.addWidget(align_widget)

        button.clicked.connect(action)

        if tooltip is not None:
            button.setToolTip(tooltip)

        if fixed_width is None:
            fixed_width = 200
        if fixed_height is None:
            fixed_height = 30
        button.setFixedHeight(fixed_height)
        button.setFixedWidth(fixed_width)

        if hotkey is not None:
            button.setShortcut(hotkey)

        if mode is not None:
            self.mode_to_btn[mode] = button

        return button

    @staticmethod
    def add_settings_button(cb, widget, tooltip=None, icon=None):
        button = QPushButton()
        button.setStyleSheet(
            """
                QPushButton {background:rgb(250,240,230); border: none;}
                QPushButton::pressed{background:rgb(117, 218, 255); border: none;}
            """
        )
        widget.addWidget(button)
        if cb is not None:
            button.clicked.connect(cb)
        if tooltip is not None:
            button.setToolTip(tooltip)

        button.setFixedHeight(50)
        button.setFixedWidth(50)
        if icon is not None:
            button.setIcon(QIcon(os.path.join(get_root_dir(), ICONS_FOLDER, icon)))
            button.setIconSize(QSize(40, 40))

        return button

    def toggle_tips(self, show):
        text = (
            " B: zoom in brush \n Shift B: zoom out brush \n"
            " In polygon mode B/Shift B are \n responsible for polygon start. \n"
            " Ctrl +: zoom in image \n Ctrl -:  zoom out image \n Ctrl Shift S: resize to fit \n Ctrl N: nullify settings sliders. \n"
            " 1..0 - layers 1 to 10 \n Alt 1..0 - layers 11 to 20 \n"
            " Ctrl F: open finder \n J: certify layer \n Ctrl H: hard example \n \n"
            "Host mode: \n # - download file \n ! - upload file"
        )
        if show:
            self.tips.setText(text)
        else:
            self.tips.setText("")

    def add_toolbar_check(
        self, cb, main_layout, text=None, initial_checked=False, hotkey=None, tooltip=None, return_checkbox=False
    ):
        wrap_widget = QWidget()
        layout = QHBoxLayout(wrap_widget)
        checkbox = QCheckBox(self)
        if text is not None:
            checkbox.setText(text)

        checkbox.setFont(default_font)
        if initial_checked:
            checkbox.setCheckState(Qt.Checked)

        if tooltip is not None:
            checkbox.setToolTip(tooltip)

        def pass_check():
            cb(checkbox.isChecked())

        checkbox.stateChanged.connect(pass_check)
        layout.addWidget(checkbox)
        main_layout.addWidget(wrap_widget)

        if hotkey is not None:
            shortcut = QShortcut(QKeySequence(hotkey), self)

            def cbtoggle():
                if not checkbox.isEnabled():
                    return
                if checkbox.isChecked():
                    checkbox.setCheckState(Qt.Unchecked)
                else:
                    checkbox.setCheckState(Qt.Checked)

            shortcut.activated.connect(cbtoggle)
            self.hotkeys_list.append(shortcut)

        if return_checkbox:
            return checkbox

    def add_menubar_check(self, text, cb, initial_checked=False, menubar=None):
        if menubar is None:
            menubar = self.view
        action = QAction(text, self)
        action.setCheckable(True)
        if initial_checked:
            action.setChecked(True)

        def pass_check():
            cb(action.isChecked())

        action.triggered.connect(pass_check)

        menubar.addAction(action)

    def add_layer_checkbox(self, cb, main_layout):
        checkbox = QCheckBox(self)

        def pass_check():
            cb(checkbox.isChecked())

        checkbox.stateChanged.connect(pass_check)
        main_layout.addWidget(checkbox)
        checkbox.setEnabled(False)

        return checkbox

    def add_toolbar_radio(self, text, cb=None, group=None, widget=None, initial_checked=False, hotkey=None, space=None):
        radio = QRadioButton(text)
        radio.setFont(default_font)
        if space is not None:
            radio.setStyleSheet(f"QRadioButton {{ spacing: {space}px; }}")

        if group is not None:
            group.addButton(radio)

        if initial_checked:
            radio.setChecked(Qt.Checked)

        if widget is None:
            widget = self.main_toolbar
        widget.addWidget(radio)

        # Only run the user supplied callback when button becomes checked. If we need to notify on
        # uncheck we can add that ability. For now it is not needed.
        if cb is not None:

            def run_cb():
                if radio.isChecked() is True:
                    cb()

            radio.toggled.connect(run_cb)

        if hotkey is not None:
            shortcut = QShortcut(QKeySequence(hotkey), self)
            f = functools.partial(radio.setChecked, Qt.Checked)
            shortcut.activated.connect(f)
            self.hotkeys_list.append(shortcut)

    @staticmethod
    def add_navigation_slider(cb):
        slider = NavigationSlider(Qt.Horizontal)
        slider.setSingleStep(1)
        slider.setRange(0, 0)
        slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        slider.setTickPosition(QSlider.TicksBelow)

        value_label = QLabel()
        value_label.setFont(default_font)

        def moved():
            value_label.setText("%.2d" % slider.sliderPosition())

        def pass_slide():
            value_label.setText("%.2d" % slider.sliderPosition())
            cb(slider.sliderPosition())

        slider.valueChanged.connect(moved)
        slider.sliderReleased.connect(pass_slide)

        return value_label, slider

    def add_toolbar_slider(
        self, cb, val, text=None, nosep=False, min_range=0, max_range=64, toolbar=None, tooltip=None
    ):
        if toolbar is None:
            toolbar = self.left_toolbar
        if text is not None:
            label = QLabel()
            label.setFont(default_font)
            label.setText(text)
            wrap_widget = self.align_widget_center(label)
            if tooltip is not None:
                wrap_widget.setToolTip(tooltip)
            toolbar.addWidget(wrap_widget)
            toolbar.addWidget(label)

        widget = QWidget(toolbar)
        slider = QSlider(Qt.Horizontal)
        slider.setTickInterval(10)
        slider.setSingleStep(1)
        slider.setRange(min_range, max_range)
        slider.setValue(val)

        value_label = QLabel()
        value_label.setFont(default_font)
        value_label.setText("%.2d" % val)

        layout = QHBoxLayout(widget)
        layout.addWidget(slider)
        layout.addWidget(value_label)
        toolbar.addWidget(widget)

        def pass_slide():
            value_label.setText("%.2d" % slider.sliderPosition())
            cb(slider.sliderPosition())

        slider.valueChanged.connect(pass_slide)
        if not nosep:
            toolbar.addSeparator()

        return slider

    def add_toolbar_hide_button(self, toolbar, start_dir="<"):
        btn = QPushButton(start_dir, self)
        btn.setMaximumWidth(30)
        btn.setMinimumHeight(100)

        def toggle():
            toolbar.toggleViewAction().setChecked(btn.text() == start_dir)
            toolbar.toggleViewAction().trigger()
            btn.setText(">" if btn.text() == "<" else "<")

        btn.clicked.connect(toggle)
        return btn

    def profile(self):
        import cProfile

        self._profile = cProfile.Profile()
        self._profile.enable()

    def stop_profile(self):
        if self._profile is not None:
            self._profile.disable()
            self._profile.print_stats(sort="cumtime")
            self._profile.print_stats(sort="tottime")
            self._profile = None

    @staticmethod
    def get_empty_logs_file():
        logs_filepath = os.path.join(pathlib.Path.home(), LOGS_FILE)
        if os.path.exists(logs_filepath):
            os.remove(logs_filepath)
        return logs_filepath

    @staticmethod
    def set_up_logging():
        logger = logging.getLogger()
        logs_file = MainWindow.get_empty_logs_file()
        handler_file = logging.FileHandler(logs_file)
        handler_cmdline = logging.StreamHandler()

        formatter = logging.Formatter("%(levelname)s - %(threadName)s - %(message)s")

        handler_file.setFormatter(formatter)
        handler_cmdline.setFormatter(formatter)

        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler_file)
        logger.addHandler(handler_cmdline)

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.settings_map = {}
        self._host_urls_list = None
        self._bucket_prefix = None

        self._path: str = ""
        self._metadata = DirectoryMetadata.load(self._path)
        self._loader = functools.partial(load_image, self._metadata, "")

        self._profile = None
        self.hotkeys_list = []
        self.config_file = os.getenv(MAKANNOTATIONS_BASE_CONFIG_FILE, os.path.join(get_root_dir(), CONFIG_FILE))
        self.home_config_file = os.path.join(pathlib.Path.home(), CONFIG_FILE)
        MainWindow.set_up_logging()

        # State variables
        self._shiftj_layers = None
        self._remote_host = None
        self._no_ssl = False
        self._remote_uuid = None
        self._host_urls_index = 0
        self._waiting_remote_host = False

        self._layers_tree: Dict[str, Any] = LAYERS_DEFAULT
        self._layers = None
        self._layers_levels = None
        self.layers_colors = None
        self.layers_help = None
        self.colors_widgets = {}
        # Configs
        self.load_config()
        self.layers_tree_backwards_compat()
        self.flatten_layers()
        self.map_layers_color()
        self._current_layer = self._layers[0]

        # Image Canvas
        scroll_widget = QScrollArea()

        self.ic = ImageCanvas(self.layer_has_mask, scroll_widget)
        scroll_widget.verticalScrollBar().valueChanged.connect(self.ic.scroll_update)
        scroll_widget.horizontalScrollBar().valueChanged.connect(self.ic.scroll_update)
        self.ic.set_layers(self._layers)
        self.ic.set_layers_colors(self.layers_colors)
        self.ic.set_app(app)
        self.ic.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        scroll_widget.setWidget(self.ic)
        scroll_widget.setWidgetResizable(True)

        # Central Widget
        self.setWindowTitle("MakAnnotations")
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.central_layout = QHBoxLayout(self.central_widget)
        self.central_layout.setContentsMargins(0, 0, 0, 0)

        # Menubar
        self.menubar = self.menuBar()
        self.menubar.setNativeMenuBar(False)

        self.image_menu = self.menubar.addMenu("File")
        self.add_button("Open Image Directory", self.open_file, hotkey=ord("O"), bar=self.image_menu)
        if self._bucket_prefix is not None:
            self.add_button("Open Image S3", self.open_s3, hotkey=ord("U"), bar=self.image_menu)
        self.add_button("Previous Image", self.prev_image, hotkey=ord("P"), bar=self.image_menu)
        self.add_button("Next Image", self.next_image, hotkey=ord("N"), bar=self.image_menu)
        self.add_button("Save/Load Settings", self.save_load_settings, bar=self.image_menu)

        self.edit_menu = self.menubar.addMenu("Edit")

        # Draws
        self.drawing_menu = self.edit_menu.addMenu("Draw")

        self.add_button("Rectangle      Shift+R", self.ic.toggle_box_drawer, bar=self.drawing_menu)
        self.add_button("Seed               Shift+S", self.ic.choose_seed, bar=self.drawing_menu)

        # Undos
        self.undo_menu = self.edit_menu.addMenu("Undo")
        self.add_button("Undo                     CTRL+Z", self.ic.undo_last_operation, bar=self.undo_menu)
        self.add_button("Rectangle             CTRL+R", self.ic.undo_box, bar=self.undo_menu)
        self.add_button("Seed                     CTRL+S", self.ic.undo_last_seed, bar=self.undo_menu)
        self.add_button("Mask                     CTRL+M", self.ic.undo_mask, bar=self.undo_menu)

        # View
        self.view = self.menubar.addMenu("View")
        self.add_menubar_check("Show Mask", self.ic.set_mask_images, initial_checked=True)
        self.add_menubar_check("Only Mask", self.ic.set_mask_only)
        self.add_menubar_check("Show Depth", self.ic.set_show_depth)
        self.add_menubar_check("Tips", self.toggle_tips)

        # Toolbar
        self.algorithm_toolbar = QToolBar(self)
        self.algorithm_toolbar.setFixedWidth(230)
        self.central_layout.addWidget(self.algorithm_toolbar)
        self.algorithm_toolbar.setOrientation(Qt.Vertical)
        self.algorithm_toolbar.hide()

        self.hide_button = self.add_toolbar_hide_button(self.algorithm_toolbar)
        self.hide_button.hide()
        self.central_layout.addWidget(self.hide_button)

        # Algorithms
        self.algorithms_menu = self.menubar.addMenu("Algorithms")

        algorithms_data = {
            Algorithms.DL_AUTOMASK: ("DL automask", self.add_dl_automask),
            Algorithms.LAB_AUTOMASK: ("LAB automask", self.add_lab_automask),
            Algorithms.GRABCUT: ("GrabCut", self.add_grabcut),
            Algorithms.CLUSTERING: ("Clustering", self.add_clustering),
            Algorithms.MASK_MANIPULATION: ("Mask manipulation", self.add_mask_manipulation),
            Algorithms.MOVING_MASKS: ("Moving Masks", self.add_moving_masks),
            Algorithms.FILTERS: ("Filters", self.add_filters),
        }

        for algorithm, data in algorithms_data.items():
            text, function = data
            self.add_algorithms_button(text, function)

        # Settings toolbar
        self.drawing_toolbar = QToolBar(self)
        self.drawing_toolbar.setMinimumHeight(130)
        self.addToolBar(self.drawing_toolbar)
        self.drawing_toolbar.setOrientation(Qt.Horizontal)
        self.drawing_toolbar_layout = QHBoxLayout()

        self.add_settings_button(
            self.open_file, self.drawing_toolbar, tooltip="Open Image Directory  [O]", icon="open.png"
        )
        if self._bucket_prefix is not None:
            self.add_settings_button(
                self.open_s3, self.drawing_toolbar, tooltip="Open S3 Image   [U]", icon="s3open.png"
            )
        self.add_settings_button(
            self.prev_image, self.drawing_toolbar, tooltip="Previous Image  [P]", icon="previous.png"
        )

        self.add_settings_button(self.next_image, self.drawing_toolbar, tooltip="Next Image  [N]", icon="next.png")
        self.add_settings_button(
            self.ic.undo_last_operation, self.drawing_toolbar, tooltip="Undo  [CTRL Z]", icon="undo.png"
        )

        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(15, 0, 0, 0)
        settings_layout.setSpacing(default_button_spacing)
        settings_layout.setAlignment(Qt.AlignCenter)

        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(default_button_spacing)

        self.add_settings_button(self.ic.zoom_in, layout, tooltip="Zoom In  [CTRL +/=]", icon="zoom_in.png")

        self.add_settings_button(self.ic.zoom_out, layout, tooltip="Zoom Out  [CTRL -]", icon="zoom_out.png")
        settings_layout.addWidget(widget)

        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(default_button_spacing)

        self.increase_brush_btn = self.add_settings_button(
            self.ic.big_brush, layout, tooltip="Increase Brush Size  [Shift-B]", icon="big_brush.png"
        )

        self.reduce_brush_btn = self.add_settings_button(
            self.ic.small_brush, layout, tooltip="Reduce Brush Size  [B]", icon="small_brush.png"
        )
        settings_layout.addWidget(widget)

        self.drawing_toolbar.addWidget(settings_widget)

        self.drawing_widget = QWidget()
        self.drawing_layout = QVBoxLayout(self.drawing_widget)
        self.drawing_layout.setContentsMargins(15, 0, 0, 0)
        self.drawing_layout.setSpacing(default_button_spacing)
        self.drawing_layout.setAlignment(Qt.AlignCenter)

        self.mode_to_btn = {}

        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(default_button_spacing)

        # Flood Fill
        self.add_drawing_button(
            self.ic.toggle_flood_fill,
            layout,
            mode=ImageCanvas.Drawing.FLOOD_FILL_MODE,
            tooltip="Flood Fill [F]",
            icon="flood_fill.png",
            hotkey=ord("F"),
        )

        # Drawer
        self.add_drawing_button(
            self.ic.toggle_drawer,
            layout,
            mode=ImageCanvas.Drawing.DRAW_MODE,
            tooltip="Brush Drawer  [D]",
            icon="brush.png",
            hotkey=ord("D"),
        )
        # Eraser
        self.add_drawing_button(
            self.ic.toggle_eraser,
            layout,
            mode=ImageCanvas.Drawing.ERASER_MODE,
            tooltip="Brush Eraser  [E]",
            icon="eraser.png",
            hotkey=ord("E"),
        )

        self.drawing_layout.addWidget(widget)

        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(default_button_spacing)

        self.add_drawing_button(
            self.ic.toggle_polyline,
            layout,
            mode=ImageCanvas.Drawing.POLYLINE_MODE,
            tooltip="Poly[L]ine",
            icon="polyline.png",
            hotkey=ord("L"),
        )

        self.add_drawing_button(
            self.ic.toggle_polygon,
            layout,
            mode=ImageCanvas.Drawing.POLYGON_DRAW_MODE,
            tooltip="Polygon Drawer  [C]",
            icon="polygon_drawer.png",
        )

        self.add_drawing_button(
            self.ic.toggle_polygon_eraser,
            layout,
            mode=ImageCanvas.Drawing.POLYGON_ERASER_MODE,
            tooltip="Polygon Eraser  [Shift C]",
            icon="polygon_eraser.png",
        )

        self.drawing_layout.addWidget(widget)
        self.ic.set_mode_to_btn(self.mode_to_btn)

        self.drawing_toolbar.addWidget(self.drawing_widget)

        wrap_widget = QWidget()
        layout = QVBoxLayout(wrap_widget)
        layout.setContentsMargins(15, 0, 0, 0)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignCenter)

        self.add_toolbar_check(
            self.ic.set_mask_images, layout, text="[S]how Mask", hotkey=ord("S"), initial_checked=True
        )
        self.add_toolbar_check(self.ic.set_mask_only, layout, text="Only [M]ask", hotkey=ord("M"))

        all_layers_enabled = self.layers_colors is not None
        show_all_layers = self.add_toolbar_check(
            self.set_show_all_layers, layout, text="All Layers Mode", return_checkbox=True
        )
        show_all_layers.setEnabled(all_layers_enabled)
        self.drawing_toolbar.addWidget(wrap_widget)

        wrap_widget = QWidget()
        layout = QVBoxLayout(wrap_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignCenter)

        self.show_depth_checkbox = self.add_toolbar_check(
            self.ic.set_show_depth, layout, text="Show Depth [Z]", hotkey=ord("Z"), return_checkbox=True
        )
        self.show_depth_checkbox.setEnabled(False)
        self.inch_grid_checkbox = self.add_toolbar_check(
            self.ic.set_inch_grid, layout, text="One Inch [G]rid", hotkey=ord("G"), return_checkbox=True
        )
        self.inch_grid_checkbox.setEnabled(False)

        self.drawing_toolbar.addWidget(wrap_widget)

        self.image_grid_ui: Optional[ImageGridUI] = None
        self.add_image_grid_ui_action: Optional[QAction] = None
        self.grid_size: Optional[List[int]] = None

        # Settings
        wrap_widget = QWidget()
        layout = QVBoxLayout(wrap_widget)
        layout.addStretch(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        image_settings = []
        self.add_settings_slider(
            self.ic.set_brightness_level,
            self.ic.brightness_value,
            layout,
            image_settings,
            icon="brightness.png",
            min_range=-100,
            max_range=100,
            tooltip="Brightness",
        )

        self.add_settings_slider(
            self.ic.set_contrast_level,
            self.ic.contrast_value,
            layout,
            image_settings,
            icon="contrast.png",
            min_range=-100,
            max_range=100,
            tooltip="Contrast",
        )

        self.add_settings_slider(
            self.ic.set_hue_level,
            self.ic.hue_value,
            layout,
            image_settings,
            icon="hue.png",
            min_range=-130,
            max_range=130,
            tooltip="Hue",
        )
        self.add_settings_slider(
            self.ic.set_opacity_level,
            self.ic.mask_opacity_value,
            layout,
            icon="mask_opacity.png",
            min_range=-50,
            max_range=50,
            tooltip="Mask Opacity",
        )

        layout.setAlignment(Qt.AlignRight)
        self.ic.set_image_settings_sliders(image_settings)
        self.drawing_toolbar.addWidget(wrap_widget)

        # Toolbar
        self.main_toolbar = QToolBar(self)
        self.main_toolbar.setMovable(True)
        self.main_toolbar.setMinimumWidth(200)
        self.central_layout.addWidget(self.main_toolbar)
        self.main_toolbar.setOrientation(Qt.Vertical)

        # User Widgets
        self.central_layout.addWidget(scroll_widget)

        # Navigation toolbar
        self.navigation_toolbar = QToolBar(self)
        self.addToolBar(Qt.BottomToolBarArea, self.navigation_toolbar)
        self.navigation_toolbar.setOrientation(Qt.Horizontal)

        self.navigation_slider_label, self.navigation_slider = self.add_navigation_slider(self.navigate_to_image)

        navigation_widget = QWidget()
        layout = QHBoxLayout(navigation_widget)
        layout.addWidget(self.navigation_slider)
        layout.addWidget(self.navigation_slider_label)

        self.navigation_toolbar.addWidget(navigation_widget)

        # Status Bar
        self.status_bar_text = QLabel()
        status_bar = QStatusBar()
        status_bar.addWidget(self.status_bar_text)
        self.setStatusBar(status_bar)
        self.ic.set_status_widget(self.status_bar_text)

        # Data Structures
        self._images: List[str] = []
        self.current_image_index = 0

        # Windows
        self.settings_window = None
        self.search_window = None
        self.grid_resize_window = None

        # Layers
        self.destination_layers_grp = None

        # Filters data
        self.filters: Dict[Filters, Union[FilterUI, LayerFiltersUI]] = {
            Filters.SKIP_CERTIFIED_IMAGES: FilterUI(SkipCertifiedImagesFilter(self.navigation_slider)),
            Filters.HARD_EXAMPLE: FilterUI(HardExampleFilter(self.navigation_slider)),
            Filters.LAYER: LayerFiltersUI(
                [
                    LayerFilterUI(LayerHasMask(self.navigation_slider)),
                    LayerFilterUI(LayerHasNoMask(self.navigation_slider)),
                    LayerFilterUI(LayerCertified(self.navigation_slider)),
                    LayerFilterUI(LayerNotCertified(self.navigation_slider)),
                ]
            ),
        }
        self.filters[Filters.LAYER].set_active_layer(self._layers[0])

        """Information stored for current image: {layer: value}"""
        self._layers_has_mask: Dict[str:bool] = {}
        self._layers_certified: Dict[str:bool] = {}
        self._layers_hard_example: Dict[str:bool] = {}

        # Status checkboxes
        self.layers_status_checkboxes = {status: {} for status in LayersStatus}

        self.drawing_status = True
        self.certified = False
        self.moving_mask_mode = False

        # Labels in main toolbar
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel("Layers")
        label.setFont(QFont("Times", 15, QFont.Bold))
        layout.addWidget(label)

        check_label_widget = QWidget()
        check_label_layout = QHBoxLayout(check_label_widget)
        check_label_layout.setAlignment(Qt.AlignRight)

        label = QLabel("Certified")
        label.setFont(QFont("Arial", 9))
        check_label_layout.addWidget(label)

        label = QLabel("Hard Example")
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter)
        label.setMaximumWidth(40)
        label.setFont(QFont("Arial", 9))
        check_label_layout.addWidget(label)

        layout.addWidget(check_label_widget)
        self.main_toolbar.addWidget(widget)

        self.layers_grp = QButtonGroup(self)

        def cb(button):
            if self.destination_layers_grp is not None:
                for button in self.destination_layers_grp.buttons():
                    button.setEnabled(True)
                disabled_button = self.destination_layers_grp.button(self.layers_grp.checkedId())
                disabled_button.setEnabled(False)

        self.layers_grp.buttonClicked.connect(cb)

        for count, layer in enumerate(self._layers):
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.addSpacing(5)
            layout.setContentsMargins(0, 0, 0, 0)

            def set_layer_cb(layer):
                return lambda: self.set_layer(layer)

            # create label
            label = QLabel(str(count + 1))
            label.setFont(default_font)
            label.setMaximumWidth(15)
            label.setMinimumWidth(15)
            layout.addWidget(label)

            self.add_toolbar_radio(
                layer,
                set_layer_cb(layer),
                group=self.layers_grp,
                widget=layout,
                initial_checked=(count == 0),
                space=20 * self._layers_levels[count] + 5,
            )
            aligned_widget = QWidget()
            aligned_layout = QHBoxLayout(aligned_widget)
            aligned_layout.setSpacing(30)
            aligned_layout.setContentsMargins(15, 0, 30, 0)

            if self.layers_help is not None and layer in self.layers_help:
                help_widget = HelpLinkWidget(self.layers_help[layer])
                aligned_layout.addWidget(help_widget)

            if self.layers_colors is not None:
                color_widget = ColorWidget(self.layers_colors[layer])
                self.colors_widgets[layer] = color_widget
                aligned_layout.addWidget(color_widget)

            def set_layer_certified_cb(layer):
                return functools.partial(self.certify_layer, layer)

            self.layers_status_checkboxes[LayersStatus.CERTIFICATION][layer] = self.add_layer_checkbox(
                set_layer_certified_cb(layer), aligned_layout
            )

            def set_layer_hard_example(layer):
                return functools.partial(self.set_layer_hard_example, layer)

            self.layers_status_checkboxes[LayersStatus.HARD_EXAMPLE][layer] = self.add_layer_checkbox(
                set_layer_hard_example(layer), aligned_layout
            )

            layout.addWidget(aligned_widget)

            self.main_toolbar.addWidget(widget)

        # label with tips
        self.tips = QLabel()
        self.main_toolbar.addWidget(self.tips)

        # Enable all layers if able
        show_all_layers.setChecked(all_layers_enabled)
        self.set_show_all_layers(all_layers_enabled)

    @property
    def images(self) -> List[str]:
        return self._images

    @property
    def images_path(self) -> str:
        return self._path

    @property
    def layers(self) -> List[str]:
        return self._layers

    @property
    def layers_has_mask(self) -> Dict[str, bool]:
        return self._layers_has_mask

    @property
    def layers_certified(self) -> Dict[str, bool]:
        return self._layers_certified

    @property
    def layers_hard_example(self) -> Dict[str, bool]:
        return self._layers_hard_example

    def set_layer(self, layer):
        self._current_layer = layer
        self.ic.set_layer(layer, self.update_layer_information)

    def set_show_all_layers(self, show_all_layers):
        self.ic.set_show_all_layers(show_all_layers)
        self.set_mode_colors_widgets(show_all_layers)

    def set_mode_colors_widgets(self, mode):
        for layer, color in self.layers_colors.items():
            self.colors_widgets[layer].set_draw_mode(mode)

    def switch_layers(self, id):
        button = self.layers_grp.button(id)
        if button is not None:
            button.animateClick()

    def update_layer_checkbox_status(self, layers_checkboxes, set_status_function):
        for checkbox in layers_checkboxes.values():
            checkbox.setEnabled((len(self._images) > 0))

        cur_checkbox = layers_checkboxes[self._current_layer]

        status = cur_checkbox.isChecked()
        set_status_function(self._current_layer, status)

        return status

    def update_layer_information(self):
        self.update_layer_certified()
        self.update_layer_hard_example()

    def update_drawing_status(self):
        enabled = np.logical_or(self.certified, self.moving_mask_mode)
        if enabled:
            self.ic.disable_drawing()
        else:
            self.ic.enable_drawing()

        self.drawing_widget.setEnabled(not enabled)

    def update_layer_certified(self):
        self.certified = self.update_layer_checkbox_status(
            self.layers_status_checkboxes[LayersStatus.CERTIFICATION], self.ic.set_layer_certified,
        )
        self.update_drawing_status()

    def update_layer_hard_example(self):
        self.update_layer_checkbox_status(
            self.layers_status_checkboxes[LayersStatus.HARD_EXAMPLE], self.ic.set_hard_example,
        )

    def certify_layer(self, layer, checked):
        self.ic.set_layer_certified(layer, checked)

        self._layers_certified[layer] = checked

        if layer == self._current_layer:
            self.certified = checked
            self.update_drawing_status()

        for filter_ui in self.filters.values():
            filter_ui.update_image_value(
                self,
                MainWindowObserverArgs(
                    update_type=MainWindowUpdate.CERTIFIED, index=self.current_image_index, layer=layer
                ),
            )

    def select_shiftj_layers(self):
        cbd = CheckboxDialog("Select Layers For Shift-J", [layer.text() for layer in self.layers_grp.buttons()])
        self._shiftj_layers = cbd.get_checked()

    def all_layers_certify(self):
        """
        Certify all layers loaded - switch the image canvas to each layer in succession and mark
        it as certified.
        """
        if self.ic is None:
            return
        if self._shiftj_layers is None:
            self.select_shiftj_layers()
            return
        for layer in self._shiftj_layers:
            # Certify it
            self.certify_layer(layer, True)
            # Set the check
            cur_checkbox = self.layers_status_checkboxes[LayersStatus.CERTIFICATION][layer]
            cur_checkbox.setChecked(True)

    def set_layer_hard_example(self, layer, checked):
        self.ic.set_hard_example(layer, checked)
        self._layers_hard_example[layer] = checked

        for filter_ui in self.filters.values():
            filter_ui.update_image_value(
                self, MainWindowObserverArgs(update_type=MainWindowUpdate.HARD_EXAMPLE, index=self.current_image_index)
            )

    def shortcut_layer_status(self, layers_flag):
        cur_checkbox = self.layers_status_checkboxes[layers_flag][self._current_layer]
        cur_checkbox.animateClick()

    def layers_tree_backwards_compat(self):
        if isinstance(self._layers_tree, list):
            self._layers_tree = {k: {} for k in self._layers_tree}

    def flatten_layers(self):
        def add_layers(layers_subtree, level):
            for layer, children in layers_subtree.items():
                assert layer not in self._layers, f"Found duplicate layer {layer}"
                self._layers.append(layer)
                self._layers_levels.append(level)
                add_layers(children, level + 1)

        self._layers = []
        self._layers_levels = []
        add_layers(self._layers_tree, 0)

    def map_layers_color(self):
        if self.color_palette is None:
            self.color_palette = DEFAULT_COLORS_PALETTE
        self.layers_colors = {}
        for count, layer in enumerate(self.layers):
            self.layers_colors[layer] = self.color_palette[count]

    def read_config_json(self):
        with open(self.config_file, "r") as f:
            config = json.load(f, object_pairs_hook=OrderedDict)

        if os.path.isfile(self.home_config_file):
            try:
                with open(self.home_config_file, "r") as f:
                    home_config = json.load(f, object_pairs_hook=OrderedDict)
                    if home_config.get("version") == config["version"]:
                        config.update(home_config)
            except Exception as e:
                print("home config load failure, ignoring it", e)

        return config

    def load_config(self):
        config = self.read_config_json()
        self.settings_map = Settings.map_from_config(config.get("slider_settings"))
        self._host_urls_list = config.get("host_urls_list")
        self._layers_tree = config.get("layers", LAYERS_DEFAULT)
        self.color_palette = config.get("colors")
        self.layers_help = config.get("help_links")
        self._bucket_prefix = config.get("s3_bucket_prefix")

    def save_config(self):
        with open(self.config_file, "r") as f:
            config = json.load(f, object_pairs_hook=OrderedDict)

        home_config = {
            "slider_settings": Settings.map_to_config(self.settings_map),
            "host_urls_list": self._host_urls_list,
            "layers": self._layers_tree,
            "colors": self.color_palette,
            "help_links": self.layers_help,
            "s3_bucket_prefix": self._bucket_prefix,
        }

        for k in list(home_config.keys()):
            if home_config[k] == config[k]:
                del home_config[k]

        home_config["version"] = config["version"]

        update = True
        if os.path.isfile(self.home_config_file):
            try:
                with open(self.home_config_file, "r") as f:
                    home_config_old = json.load(f, object_pairs_hook=OrderedDict)
                    if home_config_old == home_config:
                        update = False
            except Exception as e:
                print("home config load failure, ignoring it", e)

        if update:
            if os.path.isfile(self.home_config_file):
                os.rename(self.home_config_file, f"{self.home_config_file}.bak")

            with open(self.home_config_file, "w") as f:
                json.dump(home_config, f, indent=4)

            print(f"Updated {self.home_config_file}.")
            print(f"> Run 'diff {self.home_config_file}.bak {self.home_config_file}' to see differences.")

    def open_file(self):
        path = QFileDialog.getExistingDirectory(
            self, "Open Images Directory", directory=os.getenv("HOME", ".") + "/code/labels"
        )
        if path is not None:
            self.load_images(path)

    def open_s3(self):
        ofd = S3ListDialog(self._bucket_prefix)
        path = ofd.getSelection()
        if path is not None:
            self.load_s3_images(path)

    def load_path(self, path):
        if os.path.isfile(path):
            self.load_images(os.path.dirname(path))
            self.navigate_to_image_by_name(os.path.basename(path))
        elif os.path.isdir(path):
            self.load_images(path)

    def load_s3_images(self, path):
        if path is None or len(path) == 0:
            return

        self._path = path
        self._metadata = DirectoryMetadata.load(self._path)
        self._loader = functools.partial(load_image, self._metadata, path)

        images = get_file_listing(path)
        self._images = []
        for ext in EXTENSIONS:
            self._images.extend(
                [img for img in images if img.endswith(ext) and self._metadata.get_image_metadata(img) is not None]
            )
        self._images.sort()

        self.loaded_images_update()

    def load_images(self, path):
        if path is None or len(path) == 0:
            return

        self._path = path
        self._metadata = DirectoryMetadata.load(self._path)
        self._loader = functools.partial(load_image, self._metadata, path)

        masks = set()
        masks_extensions = ["*.mask_{}.png".format(layer) for layer in self._layers]
        for masks_ext in masks_extensions:
            masks |= set([os.path.basename(p) for p in glob.glob(os.path.join(path, masks_ext))])

        self._images = set()
        for ext in EXTENSIONS:
            self._images |= set(
                [
                    os.path.basename(p)
                    for p in glob.glob(os.path.join(path, "*" + ext))
                    if self._metadata.get_image_metadata(os.path.basename(p)) is not None
                ]
            )

        self._images = self._images.difference(masks)
        self._images = list(self._images)
        self._images.sort()

        self.loaded_images_update()

    def loaded_images_update(self):
        self.current_image_index = 0

        self.navigation_slider.setRange(1, len(self._images))
        self.navigation_slider.setValue(1)

        self.update_filters_cache()
        self.update_current_image()

    def update_current_image(self):
        if len(self._images) == 0:
            return
        image_name = os.path.basename(self._images[self.current_image_index])
        self.update_image(image_name)

    def update_image(self, image_name):
        image_meta = self._metadata.get_image_metadata(image_name)
        assert image_meta is not None
        self.add_image_grid_ui(image_meta.grid_size)
        ppi = image_meta.ppi
        if ppi is None:
            self.inch_grid_checkbox.setChecked(False)
            self.inch_grid_checkbox.setEnabled(False)
        else:
            self.inch_grid_checkbox.setEnabled(True)
        image_title = f"# {self.current_image_index + 1} {image_name}"
        self.setWindowTitle(image_title)
        self.ic.load_image(self._images[self.current_image_index], self._loader, self._path, ppi)

        if not self.ic.has_depth:
            self.show_depth_checkbox.setChecked(False)
            self.show_depth_checkbox.setEnabled(False)
        else:
            self.show_depth_checkbox.setEnabled(True)

        self.update_layers_status()
        self.update_layer_information()

    def add_image_grid_ui(self, new_grid_size: List[int]):
        if self.grid_size != new_grid_size:
            if self.add_image_grid_ui_action is not None:
                self.ic.set_normalized_image_roi([0, 0, 1, 1])
                self.drawing_toolbar.removeAction(self.add_image_grid_ui_action)
                self.image_grid_ui = None
                self.add_image_grid_ui_action = None
            if new_grid_size is not None:
                rows: int = new_grid_size[0]
                columns: int = new_grid_size[1]
                self.image_grid_ui = ImageGridUI(self.ic.set_normalized_image_roi, rows, columns)
                self.add_image_grid_ui_action = self.drawing_toolbar.addWidget(self.image_grid_ui)
            self.grid_size = new_grid_size

    def update_layers_status(self):
        self.ic.update_certification_layers(
            self.layers_status_checkboxes[LayersStatus.CERTIFICATION], self._layers_certified
        )
        self.ic.update_hard_example_layers(
            self.layers_status_checkboxes[LayersStatus.HARD_EXAMPLE], self._layers_hard_example
        )
        self.ic.update_has_mask_layers(self._layers_has_mask)

    def navigate_to_image_by_search(self, image_ref):
        try:
            index = int(image_ref)

            self.current_image_index = index - 1
            self.navigation_slider.setValue(index)
            self.update_current_image()
        except (ValueError, IndexError):
            self.navigate_to_image_by_name(image_ref)

    def is_valid_image(self, image_ref):
        if self.search_image(image_ref):
            return True
        try:
            index = int(image_ref) - 1
            if index >= 0 and index < len(self.images):
                return True
        except ValueError:
            pass

        return False

    def navigate_to_image_by_name(self, image_name):
        if image_name not in self._images:
            return
        index = self._images.index(image_name)
        # slider uses human numeration
        self.current_image_index = index
        self.navigation_slider.setValue(index + 1)
        self.update_current_image()

    def search_image(self, image_name):
        return image_name in self._images

    def navigate_to_image(self, human_index):
        index = human_index - 1
        if self.skip_image(index):
            current_prev_index = index
            current_next_index = index
            while True:
                if current_prev_index != 0:
                    current_prev_index, skip_image = self.prev_valid_image(current_prev_index, index)
                    if not skip_image:
                        self.current_image_index = current_prev_index
                        break

                if current_next_index != len(self._images) - 1:
                    current_next_index, skip_image = self.next_valid_image(current_next_index, index)
                    if not skip_image:
                        self.current_image_index = current_next_index
                        break

                if current_prev_index == 0 and current_next_index == len(self._images) - 1:
                    break

            self.navigation_slider.setValue(self.current_image_index + 1)
        else:
            self.current_image_index = index

        self.update_current_image()

    def layer_has_mask(self, layer: str, has_mask: bool):
        if self._layers_has_mask[layer] == has_mask:
            return
        self._layers_has_mask[layer] = has_mask
        for filter_ui in self.filters.values():
            filter_ui.update_image_value(
                self,
                MainWindowObserverArgs(update_type=MainWindowUpdate.MASK, index=self.current_image_index, layer=layer),
            )

    def skip_image(self, current_index, original_position=None):
        if original_position is not None and current_index == original_position:
            return False

        return any([filter_ui.skip_image(current_index) for filter_ui in self.filters.values()])

    def next_valid_image(self, current_index, original_position):
        current_index += 1
        if current_index >= len(self._images):
            current_index = 0
        return current_index, self.skip_image(current_index, original_position)

    def next_image(self):
        original_position = self.current_image_index
        while True:
            self.current_image_index, skip_image = self.next_valid_image(self.current_image_index, original_position)
            if not skip_image:
                break
        self.update_current_image()
        self.navigation_slider.setValue(self.current_image_index + 1)

    def prev_valid_image(self, current_index, original_position):
        if current_index == 0:
            current_index = len(self._images) - 1
        else:
            current_index -= 1
        return current_index, self.skip_image(current_index, original_position)

    def prev_image(self):
        if len(self._images) == 0:
            return
        original_position = self.current_image_index
        while True:
            self.current_image_index, skip_image = self.prev_valid_image(self.current_image_index, original_position)
            if not skip_image:
                break
        self.update_current_image()
        self.navigation_slider.setValue(self.current_image_index + 1)

    # Update cache
    def update_filters_cache(self):
        for filter_ui in self.filters.values():
            filter_ui.set_images(self)

    def remote_file(self):
        if self._remote_host is not None:
            self.remote_image_mask(self._remote_host, self._no_ssl)
        else:
            print("no host for --remote-host")

    def remote_mask_upload(self):
        if self._remote_host is not None:
            self.remote_mask_put(self._remote_host, self._no_ssl)
        else:
            print("no host for --remote-host")

    def get_remote_urls(self):
        if self._host_urls_list is None:
            return None, None
        if len(self._host_urls_list) <= self._host_urls_index:
            return None, None
        return (
            self._host_urls_list[self._host_urls_index]["img_url"],
            self._host_urls_list[self._host_urls_index]["mask_url"],
        )

    def remote_mask_put(self, remote_host, no_ssl=False):
        if self.ic is not None and self.ic.img is not None:
            _, mask_url = self.get_remote_urls()
            if mask_url is not None:
                sslcontext = ssl.SSLContext(ssl.PROTOCOL_SSLv23)

                mask = self.ic.img.mask(None).astype("uint8").tolist()
                mask = json.dumps(mask)
                mask = base64.b64encode(zlib.compress(mask.encode("utf-8"))).decode("ascii")
                body = {"uuid": self._remote_uuid, "mask": mask}
                data = bytes(json.dumps(body), "utf-8")
                try:
                    if no_ssl:
                        urllib.request.urlopen("http://" + remote_host + mask_url, data=data)
                    else:
                        urllib.request.urlopen("https://" + remote_host + mask_url, data=data, context=sslcontext)
                except Exception as e:
                    print("Failed to put remote masks: {}".format(e))
            else:
                print("no mask url for {}".format(self._host_urls_index))
        else:
            print("No mask to push")

    def remote_image_mask(self, remote_host, no_ssl=False):
        img_url, mask_url = self.get_remote_urls()
        if img_url is None:
            print("No image url for {}".format(self._host_urls_index))
            return
        mask = None
        if self._path:
            path = self._path + "/"
        else:
            path = ""
        try:
            # download the image
            if no_ssl:
                request = urllib.request.urlopen("http://" + remote_host + img_url)
            else:
                sslcontext = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
                request = urllib.request.urlopen("https://" + remote_host + img_url, context=sslcontext)
            filename = timestamp_filename("makannotations_")
            filepath = path + filename + ".png"
            with open(filepath, "wb") as f:
                f.write(request.read())
            self._images.append(filepath)

            # download the mask
            if no_ssl:
                request = urllib.request.urlopen("http://" + remote_host + mask_url)
            else:
                request = urllib.request.urlopen("https://" + remote_host + mask_url, context=sslcontext)
            mask = request.read().decode("utf-8")
            body = json.loads(mask)
            self._remote_uuid = body["uuid"]
            mask = body["mask"]

            self.current_image_index = len(self._images) - 1
            self.update_filters_cache()
            self.update_current_image()

        except Exception as e:
            print("Failed to get remote image: {}".format(e))
            return

        try:
            if mask is not None:
                mask = np.asarray(body["mask"]).astype(np.bool)
                self.ic.img._set_mask(mask)
                self.ic.img._update_mask()
                self.ic._draw_current()
        except Exception as e:
            print("Failed to apply remote mask: {}".format(e))

    def save_load_settings(self):
        if self.settings_window is None:
            self.settings_window = SettingsWindow(self, self.ic.get_settings, self.ic.set_settings, self.settings_map)
            self.settings_window.setFixedSize(400, 400)
            self.settings_window.show()
        else:
            self.settings_window.close()
            self.settings_window = None

    def settings_window_closed(self):
        self.settings_window = None

    def search_window_show(self):
        self.search_window = SearchWindow(self.navigate_to_image_by_search, self.is_valid_image)
        self.search_window.show()

    def grid_resize_window_show(self):
        self.grid_resize_window = ImageGridSizeSetter(self.add_image_grid_ui)
        self.grid_resize_window.show()

    def keyPressEvent(self, evt):
        modifiers = evt.modifiers()
        if modifiers == (Qt.ShiftModifier | Qt.ControlModifier):
            if evt.key() == ord("S"):
                self.ic.auto_size_image()
        elif modifiers == Qt.ControlModifier:
            if evt.key() == ord("="):
                self.ic.zoom_in()
            elif evt.key() == ord("-"):
                self.ic.zoom_out()
            elif evt.key() == ord("Z"):
                self.ic.undo_last_operation()
            elif evt.key() == ord("M"):
                self.ic.undo_mask()
            elif evt.key() == ord("A"):
                self.ic.delete_all_masks()
            elif evt.key() == ord("S"):
                self.ic.undo_last_seed()
            elif evt.key() == ord("R"):
                self.ic.undo_box()
            elif evt.key() == ord("F"):
                self.search_window_show()
            elif evt.key() == ord("H"):
                self.shortcut_layer_status(LayersStatus.HARD_EXAMPLE)
            elif evt.key() == ord("N"):
                self.ic.nullify_image_settings()
        elif modifiers == Qt.ShiftModifier:
            if evt.key() == ord("R"):
                self.ic.toggle_box_drawer()
            elif evt.key() == ord("S"):
                self.ic.choose_seed()
            elif evt.key() == ord("B"):
                self.ic.big_brush()
            elif evt.key() == ord("C"):
                self.ic.toggle_polygon_eraser()
            elif evt.key() == ord("?"):
                self.profile()
            elif evt.key() == ord("P"):
                if self.image_grid_ui is not None:
                    self.image_grid_ui.go_left()
            elif evt.key() == ord("N"):
                if self.image_grid_ui is not None:
                    self.image_grid_ui.go_right()
            elif evt.key() == ord("J"):
                self.all_layers_certify()
            elif evt.key() == ord("G"):
                if self.image_grid_ui is not None:
                    self.grid_resize_window_show()
            elif evt.key() == ord("#"):
                self.remote_file()
            elif evt.key() == ord("!"):
                self.remote_mask_upload()
        elif evt.key() == ord("B"):
            self.ic.small_brush()
        elif evt.key() == ord("C"):
            self.ic.toggle_polygon()
        elif evt.key() == ord("/"):
            self.stop_profile()
        elif evt.key() == ord("J"):
            self.shortcut_layer_status(LayersStatus.CERTIFICATION)

        #
        # 'H' to enter host entries selection mode
        # 0-9 to select the host entry and exit host entries selection mode
        # '#' to pull host entry image and mask
        # '!' to push host entry mask
        elif evt.key() == ord("H"):
            self.status_bar_text.setText("H Mode, waiting for H number ....")
            self._waiting_remote_host = True
            self.disable_hotkeys()
        elif evt.key() >= ord("0") and evt.key() <= ord("9"):
            if self._waiting_remote_host:
                self._host_urls_index = evt.key() - ord("0")
                self._waiting_remote_host = False
                self.enable_hotkeys()
                self.status_bar_text.setText("Remote host url number {}".format(self._host_urls_index))
            else:
                layer_id = (evt.key() - ord("0") + 1) * (-1)
                if evt.key() == ord("0"):
                    # Use layer=10 for 0
                    layer_id -= 10
                if modifiers == Qt.AltModifier:
                    # Add extra 10 for Alt+N
                    layer_id -= 10
                self.switch_layers(layer_id)
        elif evt.key() == Qt.Key_Escape:
            self.ic.reset_mode()

    def closeEvent(self, evt):
        if self.ic is not None:
            self.ic.process_layers_certification_data()
            self.ic.save_layer_data()

    def load_torchscript_model(self):
        path, type = QFileDialog.getOpenFileName(self, "Open TorchScript DL Model File")
        torchscript_model.load(path)

    def auto_mask_torchscript(self):
        if torchscript_model.is_loaded():
            return self.ic.auto_mask_DL(self.dl_model_channel.value())

    def process_args(self, args):
        # Command line options
        if args is not None:
            if args.open_path is not None:
                self.load_path(args.open_path)

            if args.torchscript_model is not None:
                torchscript_model.load(args.torchscript_model)

            if args.remote_host is not None:
                self._remote_host = args.remote_host
            self._no_ssl = not args.ssl
