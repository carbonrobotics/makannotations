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

import json
import os
import cv2
import math
import numpy as np
import functools
import datetime
from aenum import AutoNumberEnum
from typing import List
from PyQt5.QtCore import QRect, QSize, Qt
from PyQt5.QtGui import QColor, QCursor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QLabel

from file_io import file_exists
import torchscript_model
from mask_image import MaskImage, DEFAULT_MASK_COLOR
from metadata import CertificationData, DEFAULT_CERTIFICATION_VERSION

ZOOM_FACTOR = 0.1
BRUSH_ZOOM_FACTOR = 0.5
LABEL_PIXMAP_YOFFSET = 0
LABEL_PIXMAP_XOFFSET = 0
DEFAULT_BRUSH_RADIUS = 10
DEFAULT_PEN_START_RADIUS = 10
DEFAULT_POLYLINE_THICKNESS = 4

DEFAULT_CLOSING_ITERATIONS = 0
DEFAULT_REMOVING_OBJECTS_SIZE = 0
DEFAULT_DILATION_ITERATIONS = 0
DEFAULT_EROSION_ITERATIONS = 0
DEFAULT_CLUSTERS_AMOUNT = 10

SCROLL_PADDING = 2


class Settings:
    class Sliders(AutoNumberEnum):
        CLUSTERING = ()
        REMOVING_OBJECTS = ()
        CLOSING = ()
        DILATION = ()
        EROSION = ()

    @staticmethod
    def map_from_config(config):
        settings_map = {}
        for k, v in config.items():
            settings_map[k] = Settings.from_map(v)
        return settings_map

    @staticmethod
    def map_to_config(settings_map):
        config = {}
        for k, v in settings_map.items():
            config[k] = v.to_map()
        return config

    @staticmethod
    def from_map(instance_map):
        obj = Settings()
        variables = vars(obj).keys()
        for var in variables:
            setattr(obj, var, instance_map.get(var, getattr(obj, var)))
        return obj

    def __init__(self):
        self.algo_stack = ""

        self.clusters_amount = DEFAULT_CLUSTERS_AMOUNT
        self.removing_objects_size = DEFAULT_REMOVING_OBJECTS_SIZE
        self.closing_iterations_image = DEFAULT_CLOSING_ITERATIONS
        self.dilation_iterations = DEFAULT_DILATION_ITERATIONS
        self.erosion_iterations = DEFAULT_EROSION_ITERATIONS

    def __str__(self):
        return "1: [{}, {}], 2: [{}, {}], 3: [{}, {}], " + "4: [{}, {}], 5: [{}, {}], 6: [{}, {}]".format(
            *vars(self).values()
        )

    def to_map(self):
        return vars(self)


class ImageCanvas(QLabel):
    class Drawing(AutoNumberEnum):
        NONE_MODE = ()
        ERASER_MODE = ()
        DRAW_MODE = ()
        FLOOD_FILL_MODE = ()
        BOX_MODE = ()
        SEED_MODE = ()
        POLYGON_DRAW_MODE = ()
        POLYGON_ERASER_MODE = ()
        POLYLINE_MODE = ()

    def __init__(self, layer_has_mask_callback, scrol_widget):
        super().__init__(scrol_widget)
        self.scroll_widget = scrol_widget
        self.mask_only = False
        self.show_depth = False
        self.filename = None
        self.path = None
        self.setMouseTracking(True)
        self.status_widget = None
        self.algorithms_widget = None
        self.zoom_level = 0
        self.img = None
        self.last_x = None
        self.last_y = None
        self.mask_images = True
        self.mode = ImageCanvas.Drawing.NONE_MODE
        self.last_mode = ImageCanvas.Drawing.NONE_MODE
        self.moving_mask = False
        self.app = None
        self.brush_zoom = 0
        self.layers = None
        self.layer = None
        self.mode_to_btn = None
        # pixels per inch
        self.show_inch_grid = False
        self.ppi_value = None

        # Control values
        self.settings = Settings()
        self.sliders = None
        self.box_begin = None
        self.prev_draw_x = None
        self.prev_draw_y = None

        self.contrast_value = 0
        self.brightness_value = 0
        self.hue_value = 0
        self.mask_opacity_value = 0
        self.image_settings_sliders = None

        # common data for polygon and polyline
        self.polyshape_finish_point_x = None
        self.polyshape_finish_point_y = None
        self.prev_polyshape_finish_point_x = None
        self.prev_polyshape_finish_point_y = None

        self.prev_polyshape_point_x = None
        self.prev_polyshape_point_y = None
        self.prev_polyshape_scale = None

        self.polyshape_points_stack = None
        self.polyshape_points_stack_prev = None

        # polygon
        self.polygon_start_zoom = 0

        # visible region
        self.visible_rect = None
        self.visible_pixmap = None
        self.vis_region_x_start = None
        self.vis_region_x_end = None
        self.vis_region_y_start = None
        self.vis_region_y_end = None

        # updates flags
        self._scroll_update = False

        self._drawing_enabled = True

        self.destination_layer = None

        self.layer_has_mask_callback = layer_has_mask_callback

        self.show_all_layers = False
        self.layers_colors = None
        self.colors_masks = None
        self.mask_color = DEFAULT_MASK_COLOR

        self.layers_certified = {}
        self.layers_hard_example = {}

        # original size image region of interest
        self._normalized_image_roi: List[float] = [0, 0, 1, 1]

    @staticmethod
    def image_has_unncertified_layers(path, image_filename, layers):
        for layer in layers:
            if not ImageCanvas.layer_certified(path, image_filename, layer) and ImageCanvas.layer_has_mask(
                path, image_filename, layer
            ):
                return True
        return False

    @staticmethod
    def image_has_hard_example(path, image_filename, layers):
        for layer in layers:
            certification_filename = CertificationData.make_certification_filename(path, image_filename, layer)
            if ImageCanvas.hard_example_status(certification_filename):
                return True
        return False

    @staticmethod
    def layer_certified(path, image_filename, layer):
        certification_filename = CertificationData.make_certification_filename(path, image_filename, layer)
        return ImageCanvas.certification_status(certification_filename)

    @staticmethod
    def layer_has_mask(path, image_filename, layer):
        mask_filename = MaskImage.make_label_filename(path, image_filename, layer)
        return file_exists(mask_filename)

    def set_layer_certified(self, layer, certified):
        if self.img is None:
            return

        if layer == self.layer:
            self.img.set_mask_certified(certified)

        self.layers_certified[layer] = certified

    def set_hard_example(self, layer, hard_example):
        if self.img is None:
            return

        if layer == self.layer:
            self.img.set_hard_example(hard_example)

        self.layers_hard_example[layer] = hard_example

    def process_certification_data(self, layer):
        if self.path is None or self.filename is None:
            return
        certification_filepath = CertificationData.make_certification_filename(self.path, self.filename, layer)
        previous_certification_data = CertificationData.load(certification_filepath)
        if (
            previous_certification_data.hard_example != self.layers_hard_example[layer]
            or previous_certification_data.certified != self.layers_certified[layer]
        ):
            new_certification_data = CertificationData(
                version=DEFAULT_CERTIFICATION_VERSION,
                certified=self.layers_certified[layer],
                username=os.getenv("USER"),
                source=previous_certification_data.source,
                timestamp=datetime.datetime.utcnow().isoformat(timespec="seconds"),
                md5sum=previous_certification_data.md5sum,
                hard_example=self.layers_hard_example[layer],
            )
            new_certification_data.write(certification_filepath)

    def process_layers_certification_data(self):
        for layer in self.layers:
            self.process_certification_data(layer)

    @staticmethod
    def certification_status(certification_filename):
        certification_data = CertificationData.load(certification_filename)
        return certification_data.certified

    @staticmethod
    def hard_example_status(certification_filename):
        certification_data = CertificationData.load(certification_filename)
        return certification_data.hard_example

    def update_layer_checkboxes(self, layers_checkboxes, layers_cache, checked_status_function):
        for layer, checkbox in layers_checkboxes.items():
            certification_filename = CertificationData.make_certification_filename(
                self.path, self.filename, layer
            )
            status = checked_status_function(certification_filename)
            layers_cache[layer] = status
            checkbox.blockSignals(True)
            checkbox.setChecked(status)
            checkbox.blockSignals(False)

    def update_certification_layers(self, layers_checkboxes, layers_cache):
        self.update_layer_checkboxes(layers_checkboxes, layers_cache, ImageCanvas.certification_status)
        self.layers_certified = layers_cache.copy()

    def update_hard_example_layers(self, layers_checkboxes, layers_cache):
        self.update_layer_checkboxes(layers_checkboxes, layers_cache, ImageCanvas.hard_example_status)
        self.layers_hard_example = layers_cache.copy()

    def update_has_mask_layers(self, layers_has_mask):
        layers_has_mask.clear()
        for layer in self.layers:
            if layer == self.layer:
                layers_has_mask[layer] = self.img.mask().any()
            else:
                layers_has_mask[layer] = ImageCanvas.layer_has_mask(self.path, self.filename, layer)

    def set_mode_to_btn(self, mode_to_btn):
        self.mode_to_btn = mode_to_btn

    def get_layer(self):
        return self.layer

    def set_layers(self, layers):
        self.layers = layers
        self.layer = self.layers[0]

    def set_sliders(self, sliders):
        self.sliders = sliders

    def set_image_settings_sliders(self, sliders):
        self.image_settings_sliders = sliders

    def set_settings(self, settings):
        self.settings = settings
        self.update_slider_values()
        self.load_algo_stack_settings()

    def get_settings(self):
        self.save_algo_stack_settings()
        return self.settings

    def update_slider_values(self):
        if self.sliders is not None:
            if Settings.Sliders.CLUSTERING in self.sliders:
                self.sliders[Settings.Sliders.CLUSTERING].setValue(self.settings.clusters_amount)
            if Settings.Sliders.REMOVING_OBJECTS in self.sliders:
                self.sliders[Settings.Sliders.REMOVING_OBJECTS].setValue(self.settings.removing_objects_size)
            if Settings.Sliders.CLOSING in self.sliders:
                self.sliders[Settings.Sliders.CLOSING].setValue(self.settings.closing_iterations_image)
            if Settings.Sliders.DILATION in self.sliders:
                self.sliders[Settings.Sliders.DILATION].setValue(self.settings.dilation_iterations)
            if Settings.Sliders.EROSION:
                self.sliders[Settings.Sliders.EROSION].setValue(self.settings.erosion_iterations)

    @staticmethod
    def get_radius(radius, zoom):
        if zoom > 0:
            radius = radius * (1 + zoom * BRUSH_ZOOM_FACTOR)
        elif zoom < 0:
            radius = radius * (1.0 / (1 + abs(zoom) * BRUSH_ZOOM_FACTOR))
        return radius

    def pen_start_radius(self):
        return self.get_radius(DEFAULT_PEN_START_RADIUS, self.polygon_start_zoom)

    def brush_radius(self):
        return int(self.get_radius(DEFAULT_BRUSH_RADIUS, self.brush_zoom))

    def set_app(self, app):
        self.app = app

    def mouseMoveEvent(self, evt):
        pos = self.get_cursor_position()
        if self.mode not in [
            ImageCanvas.Drawing.POLYGON_DRAW_MODE,
            ImageCanvas.Drawing.POLYGON_ERASER_MODE,
            ImageCanvas.Drawing.POLYLINE_MODE,
            ImageCanvas.Drawing.FLOOD_FILL_MODE,
        ]:
            self.handle_button(self.app.mouseButtons())
        if pos.x() == evt.x() and pos.y() == evt.y():
            self._draw_current()

    def _reset_polyshape_data(self):
        self.polyshape_finish_point_x = None
        self.polyshape_finish_point_y = None
        self.prev_polyshape_finish_point_x = None
        self.prev_polyshape_finish_point_y = None

        self.prev_polyshape_point_x, self.prev_polyshape_point_y = None, None
        self.polygon_start_zoom = 0
        self.prev_polyshape_scale = None

        self.polyshape_points_stack = None
        self.polyshape_points_stack_prev = None

        if self.img is not None:
            self.img.reset_polygon_data()

    def _to_original_scale(self, value):
        return round(value / self.img.get_scale())

    def _to_current_scale(self, value):
        return round(value * self.img.get_scale())

    def finish_polyshape(self):
        if self.mode == ImageCanvas.Drawing.POLYLINE_MODE:
            last_x, last_y = self.last_x, self.last_y
            radius = DEFAULT_POLYLINE_THICKNESS
        else:
            last_x, last_y = self._to_original_scale(self.last_x), self._to_original_scale(self.last_y)
            radius = self._to_original_scale(self.pen_start_radius())

        return np.power(last_x - self.polyshape_finish_point_x, 2.0) + np.power(
            last_y - self.polyshape_finish_point_y, 2.0
        ) <= np.power(radius, 2.0)

    def remember_polyshape_state(self):
        self.prev_polyshape_finish_point_x = self.polyshape_finish_point_x
        self.prev_polyshape_finish_point_y = self.polyshape_finish_point_y
        self.polyshape_finish_point_x, self.polyshape_finish_point_y = None, None
        self.polyshape_points_stack_prev = self.polyshape_points_stack
        self.polyshape_points_stack = None
        self.polyshape_points_stack_prev.append((self.last_x, self.last_y, self.img.get_scale()))

    def remember_polyshape_points_data(self, x, y):
        self.prev_polyshape_point_x, self.prev_polyshape_point_y = x, y
        self.prev_polyshape_scale = self.img.get_scale()
        if self.polyshape_points_stack is not None:
            self.polyshape_points_stack.append(
                (self.prev_polyshape_point_x, self.prev_polyshape_point_y, self.prev_polyshape_scale)
            )

    def handle_button(self, button):
        if (
            self.mode == ImageCanvas.Drawing.ERASER_MODE or self.mode == ImageCanvas.Drawing.DRAW_MODE
        ) and button == Qt.LeftButton:
            assert (self.prev_draw_x is not None) == (self.prev_draw_y is not None)
            if self.mode == ImageCanvas.Drawing.ERASER_MODE:
                if self.prev_draw_x is None:
                    self.img.mask_erase(self.last_x, self.last_y, self.brush_radius())
                else:
                    self.img.erase_line(
                        (self._to_original_scale(self.prev_draw_x), self._to_original_scale(self.prev_draw_y)),
                        (self._to_original_scale(self.last_x), self._to_original_scale(self.last_y)),
                        self._to_original_scale(self.brush_radius()),
                    )
            if self.mode == ImageCanvas.Drawing.DRAW_MODE:
                if self.prev_draw_x is None:
                    self.img.mask_draw(self.last_x, self.last_y, self.brush_radius())
                else:
                    self.img.draw_line(
                        (self._to_original_scale(self.prev_draw_x), self._to_original_scale(self.prev_draw_y)),
                        (self._to_original_scale(self.last_x), self._to_original_scale(self.last_y)),
                        self._to_original_scale(self.brush_radius()),
                    )
            self.prev_draw_x = self.last_x
            self.prev_draw_y = self.last_y
        if (self.mode == ImageCanvas.Drawing.BOX_MODE or self.moving_mask) and button == Qt.LeftButton:
            if self.box_begin is None:
                self.box_begin = (self.last_x, self.last_y)
        if (
            self.mode
            in [
                ImageCanvas.Drawing.POLYGON_DRAW_MODE,
                ImageCanvas.Drawing.POLYGON_ERASER_MODE,
                ImageCanvas.Drawing.POLYLINE_MODE,
            ]
        ) and button == Qt.LeftButton:
            if self.mode == ImageCanvas.Drawing.POLYGON_DRAW_MODE:
                mode = MaskImage.Action.POLYGON_DRAW_MODE
            elif self.mode == ImageCanvas.Drawing.POLYGON_ERASER_MODE:
                mode = MaskImage.Action.POLYGON_ERASER_MODE
            elif self.mode == ImageCanvas.Drawing.POLYLINE_MODE:
                mode = MaskImage.Action.POLYLINE_MODE
            # initializing polyshape
            if self.polyshape_finish_point_x is None and self.polyshape_finish_point_y is None:
                if self.mode == ImageCanvas.Drawing.POLYLINE_MODE:
                    self.polyshape_finish_point_x = self.last_x
                    self.polyshape_finish_point_y = self.last_y
                else:
                    self.polyshape_finish_point_x = self._to_original_scale(self.last_x)
                    self.polyshape_finish_point_y = self._to_original_scale(self.last_y)
                self.polyshape_points_stack = []
                self.img.clean_undo_stack_polyshape()
                self.img.undo_stack.append(mode)
                self.remember_polyshape_points_data(self.last_x, self.last_y)
            # closing polyshape
            elif self.finish_polyshape():
                if self.mode in [ImageCanvas.Drawing.POLYGON_DRAW_MODE, ImageCanvas.Drawing.POLYGON_ERASER_MODE]:
                    if len(self.polyshape_points_stack) > 2:
                        self.last_x = self._to_current_scale(self.polyshape_finish_point_x)
                        self.last_y = self._to_current_scale(self.polyshape_finish_point_y)
                        self.img.polygon_line(
                            (self.prev_polyshape_point_x, self.prev_polyshape_point_y),
                            (self.last_x, self.last_y),
                            self.prev_polyshape_scale,
                            mode,
                        )
                        self.img.fill_polygon(mode)
                elif self.mode == ImageCanvas.Drawing.POLYLINE_MODE:
                    self.img.undo_stack.append(mode)
                self.remember_polyshape_state()
                self.remember_polyshape_points_data(self.last_x, self.last_y)
            else:
                if self.mode == ImageCanvas.Drawing.POLYLINE_MODE:
                    self.img.draw_polyline(
                        (self.prev_polyshape_point_x, self.prev_polyshape_point_y),
                        (self.last_x, self.last_y),
                        self.prev_polyshape_scale,
                        DEFAULT_POLYLINE_THICKNESS,
                    )
                    self.polyshape_finish_point_x = self.last_x
                    self.polyshape_finish_point_y = self.last_y
                else:
                    self.img.polygon_line(
                        (self.prev_polyshape_point_x, self.prev_polyshape_point_y),
                        (self.last_x, self.last_y),
                        self.prev_polyshape_scale,
                        mode,
                    )
                self.remember_polyshape_points_data(self.last_x, self.last_y)
        if self.mode == ImageCanvas.Drawing.SEED_MODE and button == Qt.LeftButton:
            self.img.seed_draw(self.last_x, self.last_y, self.brush_radius())
        if self.mode == ImageCanvas.Drawing.FLOOD_FILL_MODE and button == Qt.LeftButton:
            self.img.flood_fill(self._to_original_scale(self.last_x), self._to_original_scale(self.last_y))
        if button == Qt.RightButton:
            self.undo_mask()

    def mousePressEvent(self, evt):
        if self.img is None:
            return
        self.handle_button(evt.button())
        self._draw_current()

    def mouseReleaseEvent(self, evt):
        if self.img is None:
            return
        if (
            self.mode == self.Drawing.DRAW_MODE or self.mode == self.Drawing.ERASER_MODE
        ) and evt.button() == Qt.LeftButton:
            self.prev_draw_x = None
            self.prev_draw_y = None
        if (
            (self.mode == ImageCanvas.Drawing.BOX_MODE or self.moving_mask)
            and evt.button() == Qt.LeftButton
            and self.box_begin is not None
        ):
            current_box = (self.box_begin[0], self.box_begin[1], self.last_x, self.last_y)
            if self.moving_mask:
                mode = MaskImage.Action.MASK_MOVE
            else:
                mode = MaskImage.Action.BOX_MODE
            self.img.box_draw(current_box, mode=mode)
            self.box_begin = None

        self._draw_current()

    def _redraw_algorithms_stack(self):
        if self.algorithms_widget is not None:
            algo_stack = self.img.get_algo_stack()
            algo_stack = [(str(algo), str(value)) for algo, value in algo_stack]
            algo_text = []
            for algo, value in algo_stack:
                if algo == "Algorithm.AUTO_MASK_DL":
                    algo_text.append("DL mask: channel {}".format(value))
                elif algo == "Algorithm.LAB_AUTO_MASK":
                    algo_text.append("LAB_automask")
                elif algo == "Algorithm.CLUSTERING":
                    algo_text.append("Clustering: {} amount of clusters".format(value))
                elif algo == "Algorithm.BRIGHT_AUTO_MASK":
                    algo_text.append("Bright_auto_mask")
                elif algo == "Algorithm.DILATION":
                    algo_text.append("Dilation: {}".format(value))
                elif algo == "Algorithm.EROSION":
                    algo_text.append("Erosion: {}".format(value))
                elif algo == "Algorithm.REMOVING_OBJECTS":
                    algo_text.append("Removing_objects_size: {}".format(value))
                elif algo == "Algorithm.CLOSING_ITERATIONS":
                    algo_text.append("Closing_iterations: {}".format(value))
            algo_text_res = "\n".join(algo_text)
            self.algorithms_widget.setText(algo_text_res)

    def _redraw_status(self):
        if self.status_widget is not None and self.last_x is not None and self.last_y is not None:
            x = self.last_x
            y = self.last_y
            color_text = ""
            # Grab colors from appropriately sized but pre-mask reference image
            x_origin, y_origin = x, y
            if self.visible_rect is not None:
                x = x - self.vis_region_x_start
                y = y - self.vis_region_y_start
                if self.img is not None and 0 <= x < self.img.display_width() and 0 <= y < self.img.display_height():
                    hsvimg = self.img.get_display_hsv()
                    hugh = hsvimg[y, x, 0]
                    saturation = hsvimg[y, x, 1]
                    value = hsvimg[y, x, 2]
                    rgbimg = self.img.get_display_rgb()
                    red = rgbimg[y, x, 0]
                    green = rgbimg[y, x, 1]
                    blue = rgbimg[y, x, 2]
                    labimg = self.img.get_display_lab()
                    lumin = labimg[y, x, 0]
                    acolor = labimg[y, x, 1]
                    bcolor = labimg[y, x, 2]
                    color_text = "  [R: %d, G: %d, B: %d], [H: %d, S: %d, V: %d] [L: %d, A: %d, B: %d]" % (
                        red,
                        green,
                        blue,
                        hugh,
                        saturation,
                        value,
                        lumin,
                        acolor,
                        bcolor,
                    )

            self.status_widget.setText(
                "Zoom: {} Mouse {}, {}{}".format(self.zoom_level, x_origin, y_origin, color_text)
            )

    def scaling_and_drawing_polyshape_line(self, pen):
        # scaled according to previous scale
        scale = self.img.get_scale() / self.prev_polyshape_scale
        x_start = round(self.prev_polyshape_point_x * scale)
        y_start = round(self.prev_polyshape_point_y * scale)
        x_end, y_end = self.last_x, self.last_y
        parameters = x_start, y_start, x_end, y_end

        self.activate_painter(pen, "line", parameters)
        return x_start, y_start

    def activate_painter(self, pen, mode, parameters):
        painter = QPainter()
        painter.begin(self)
        painter.setPen(pen)
        if mode == "elipse":
            x, y, semi_axis, semi_axis = parameters
            painter.drawEllipse(x, y, semi_axis, semi_axis)
        if mode == "line":
            x_start, y_start, x_end, y_end = parameters
            painter.drawLine(x_start, y_start, x_end, y_end)
        if mode == "rect":
            x_start, y_start, x_end, y_end = parameters
            painter.drawRect(x_start, y_start, x_end, y_end)

        painter.end()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.visible_pixmap is not None:
            painter = QPainter()
            painter.begin(self)
            painter.drawPixmap(self.visible_rect, self.visible_pixmap)
            painter.end()

        if self.last_x is not None and self.last_y is not None:
            color = None
            if self.mode == ImageCanvas.Drawing.ERASER_MODE:
                color = [255, 0, 0]
            elif self.mode == ImageCanvas.Drawing.DRAW_MODE:
                color = [64, 255, 64]
            elif self.mode == ImageCanvas.Drawing.SEED_MODE:
                color = [90, 180, 90]
            elif (self.mode == ImageCanvas.Drawing.BOX_MODE or self.moving_mask) and self.box_begin is not None:
                color = [23, 3, 89]

                pen = QPen(Qt.black, 3, Qt.SolidLine)
                x, y = self.box_begin
                w, h = self.last_x - x, self.last_y - y
                parameters = x, y, w, h
                self.activate_painter(pen, "rect", parameters)
            elif (
                self.mode in [ImageCanvas.Drawing.POLYGON_DRAW_MODE, ImageCanvas.Drawing.POLYGON_ERASER_MODE]
                and self.polyshape_finish_point_x is not None
            ):
                # line
                if self.mode == ImageCanvas.Drawing.POLYGON_DRAW_MODE:
                    pen = QPen(QColor(*self.mask_color[:3]), 2, Qt.DashDotLine)
                elif self.mode == ImageCanvas.Drawing.POLYGON_ERASER_MODE:
                    pen = QPen(Qt.black, 2, Qt.DashDotLine)

                self.scaling_and_drawing_polyshape_line(pen)

                # circle
                color_start = [47, 86, 223]
                pen = QPen(QColor(*color_start), 2, Qt.SolidLine)
                # parameters
                radius = self.pen_start_radius()
                x = self._to_current_scale(self.polyshape_finish_point_x) - radius
                y = self._to_current_scale(self.polyshape_finish_point_y) - radius
                parameters = x, y, radius * 2, radius * 2
                self.activate_painter(pen, "elipse", parameters)
            elif self.mode == ImageCanvas.Drawing.POLYLINE_MODE:
                thickness = round(DEFAULT_POLYLINE_THICKNESS * self.img.get_scale())
                if self.polyshape_finish_point_x is None:
                    # circle
                    x_start = self.last_x
                    y_start = self.last_y
                else:
                    pen = QPen(QColor(*self.mask_color[:3], 100), 2 * thickness, Qt.SolidLine)
                    pen.setCapStyle(Qt.RoundCap)

                    x_start, y_start = self.scaling_and_drawing_polyshape_line(pen)

                # circle
                pen = QPen(Qt.black, 2, Qt.SolidLine)
                # parameters
                radius = thickness
                x = x_start - radius
                y = y_start - radius
                parameters = x, y, radius * 2, radius * 2
                self.activate_painter(pen, "elipse", parameters)
            elif self.mode == ImageCanvas.Drawing.FLOOD_FILL_MODE:
                pen = QPen(QColor(*self.mask_color[:3]), 4)
                radius = 2
                x = self.last_x - radius
                y = self.last_y - radius
                parameters = x, y, radius * 2, radius * 2
                self.activate_painter(pen, "elipse", parameters)
            if color is not None and not (self.mode == ImageCanvas.Drawing.BOX_MODE or self.moving_mask):
                pen = QPen(QColor(*color), 2, Qt.SolidLine)
                x = self.last_x - self.brush_radius()
                y = self.last_y - self.brush_radius()
                parameters = x, y, self.brush_radius() * 2, self.brush_radius() * 2

                self.activate_painter(pen, "elipse", parameters)

    def set_status_widget(self, sw):
        self.status_widget = sw

    def set_algorithms_stack_widget(self, al_text):
        self.algorithms_widget = al_text

    def set_mask_images(self, mask):
        self.mask_images = mask
        self.perform_action_on_image_and_redraw(lambda: self.img.apply_mask(mask))

    def set_mask_only(self, only):
        self.mask_only = only
        self.perform_action_on_image_and_redraw(lambda: self.img.only_mask(only))

    def set_show_depth(self, value):
        self.show_depth = value
        self.perform_action_on_image_and_redraw(lambda: self.img.set_show_depth(value))

    def set_inch_grid(self, show_grid):
        self.show_inch_grid = show_grid
        self.perform_action_on_image_and_redraw(lambda: self.img.apply_inch_grid(show_grid))

    def set_layers_colors(self, layers_colors):
        self.layers_colors = layers_colors

    def set_show_all_layers(self, show_all_layers):
        self.show_all_layers = show_all_layers
        self.perform_action_on_image_and_redraw(lambda: self.apply_show_all_layers())

    def apply_show_all_layers(self):
        self.update_colors_masks_information()
        self.img.set_colors_masks(self.colors_masks)
        self.img.set_mask_color(self.mask_color)
        self.img.set_show_all_layers(self.show_all_layers)

    def update_colors_masks_information(self):
        self.colors_masks = None
        self.mask_color = DEFAULT_MASK_COLOR
        if self.show_all_layers:
            self.make_colors_masks()

    def make_colors_masks(self):
        if self.layers_colors is None:
            return
        self.colors_masks = []
        for layer, color in self.layers_colors.items():
            if layer == self.layer:
                self.mask_color = color
            else:
                mask = self.get_mask(layer)
                if mask is not None:
                    self.colors_masks.append([color, mask.astype(np.bool)])

    def get_mask(self, layer):
        label_file = MaskImage.make_label_filename(self.path, self.filename, layer)
        mask = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
        return mask

    def load_image(self, filename, loader, path, ppi_value=None):
        if self.filename is not None:
            self.process_layers_certification_data()
        self.filename = filename
        self.path = path
        self.ppi_value = ppi_value
        self.save_layer_data()
        self.update_colors_masks_information()
        img = MaskImage.from_file(
            self.filename,
            loader,
            path,
            self.mask_images,
            self.mask_only,
            self.show_depth,
            self.layer,
            self.show_inch_grid,
            self.ppi_value,
            self.destination_layer,
            self.mask_opacity_value,
            functools.partial(self.layer_has_mask_callback, self.layer),
            self.layer_has_mask_callback,
            self.apply_show_all_layers,
            self.contrast_value,
            self.brightness_value,
            self.hue_value,
            self.show_all_layers,
            self.colors_masks,
            self.mask_color,
            self._normalized_image_roi,
        )
        if img is None:
            return
        self.set_new_image(img)

    def set_layer(self, layer, update_layer_information_callback):
        self.layer = layer
        if self.img:
            self.process_layers_certification_data()
            self.save_layer_data()
            img_data = None
            depth_data = None
            if self.img is not None:
                img_data = self.img.origin_img
                depth_data = self.img.depth
            self.update_colors_masks_information()
            img = MaskImage.from_image(
                self.filename,
                self.path,
                img_data,
                depth_data,
                self.mask_images,
                self.mask_only,
                self.show_depth,
                self.layer,
                self.show_inch_grid,
                self.ppi_value,
                self.destination_layer,
                self.mask_opacity_value,
                functools.partial(self.layer_has_mask_callback, self.layer),
                self.layer_has_mask_callback,
                self.apply_show_all_layers,
                self.contrast_value,
                self.brightness_value,
                self.hue_value,
                self.show_all_layers,
                self.colors_masks,
                self.mask_color,
                self._normalized_image_roi,
            )
            self.set_new_image(img)

        update_layer_information_callback()

    def _new_image_update(self):
        self.get_cursor_position()
        self._draw_current()

    def set_new_image(self, img: MaskImage):
        self._disable_updates()
        try:
            self.img = img
            self._set_zoom()
            self._new_image_update()
            self._reset_sliders()
            self._reset_polyshape_data()
        finally:
            self._enable_updates()

    def save_layer_data(self):
        if self.img is not None:
            self.img.save_image_mask()
            self.img.save_certification_file()

    def _cur_pixmap(self):
        # getting visible boundaries
        vis_region = self.visibleRegion()
        bound_rect = vis_region.boundingRect()
        width = bound_rect.width()
        height = bound_rect.height()
        if width == 0 or height == 0:
            return
        self.vis_region_x_start = bound_rect.x()
        self.vis_region_y_start = bound_rect.y()

        img = self.img.chopped_display(
            self._to_original_scale(self.vis_region_x_start),
            self._to_original_scale(self.vis_region_y_start),
            self._to_original_scale(width) + 1,
            self._to_original_scale(height) + 1,
        )

        self.vis_region_x_end = self.vis_region_x_start + self.img.display_width()
        self.vis_region_y_end = self.vis_region_y_start + self.img.display_height()

        bytes_per_line = self.img.display_channels() * self.img.display_width()
        qimg = QImage(
            img.data, self.img.display_width(), self.img.display_height(), bytes_per_line, QImage.Format_RGB888
        )
        self.visible_rect = QRect(
            self.vis_region_x_start, self.vis_region_y_start, self.img.display_width(), self.img.display_height()
        )

        self.visible_pixmap = QPixmap(qimg)

    @staticmethod
    def within_limits(point, low_limit, upper_limit):
        if point < low_limit:
            point = low_limit
        if upper_limit and point > upper_limit:
            point = upper_limit

        return point

    def get_cursor_position(self):
        pos = self.mapFromGlobal(QCursor.pos())
        self.last_x = pos.x() - LABEL_PIXMAP_XOFFSET
        self.last_x = self.within_limits(self.last_x, 0, self.vis_region_x_end)

        self.last_y = pos.y() - LABEL_PIXMAP_YOFFSET
        self.last_y = self.within_limits(self.last_y, 0, self.vis_region_y_end)

        return pos

    def _disable_updates(self):
        self._scroll_update = False

    def _enable_updates(self):
        self._scroll_update = True

    def scroll_update(self):
        if self._scroll_update:
            self.get_cursor_position()
            self._draw_current()

    def _draw_current(self):
        if self.img is None:
            return

        self._cur_pixmap()
        self._redraw_status()
        self.update()
        self._redraw_algorithms_stack()

    def auto_size_image(self):
        desirable_size, original_size = self._choose_dimension_for_scaling()
        self.zoom_level = ImageCanvas.find_zoom_level(desirable_size, original_size)
        self._set_zoom()

    def _choose_dimension_for_scaling(self):
        width = self.scroll_widget.width() - SCROLL_PADDING
        height = self.scroll_widget.height() - SCROLL_PADDING

        height_dif = self.img.roi_height / height
        width_dif = self.img.roi_width / width

        return (width, self.img.roi_width) if width_dif > height_dif else (height, self.img.roi_height)

    @staticmethod
    def find_zoom_level(desirable_size, original_size):
        scale = desirable_size / original_size
        if scale > 1:
            zoom_level = math.floor((scale - 1) / ZOOM_FACTOR)
        elif 0 < scale < 1:
            zoom_level = math.floor(-(1.0 / scale - 1) / ZOOM_FACTOR)

        return zoom_level

    def _set_zoom(self):
        amt = None
        if self.zoom_level > 0:
            amt = 1 + self.zoom_level * ZOOM_FACTOR
        elif self.zoom_level < 0:
            amt = 1.0 / (1 + abs(self.zoom_level) * ZOOM_FACTOR)
        else:
            amt = 1.0

        if amt is not None:
            self.img.set_scale(amt)
            self.setFixedSize(
                QSize(self._to_current_scale(self.img.roi_width), self._to_current_scale(self.img.roi_height),)
            )
            self._draw_current()

    def zoom_in(self):
        if self.img is None:
            return
        self.zoom_level += 1
        self._set_zoom()

    def zoom_out(self):
        if self.img is None:
            return
        self.zoom_level -= 1
        self._set_zoom()

    def change_btns_mode(self, checked_mode=None):
        for mode, btn in self.mode_to_btn.items():
            if mode is not checked_mode:
                btn.setChecked(False)
        if checked_mode in self.mode_to_btn.keys():
            self.mode_to_btn[checked_mode].setChecked(True)

    def disable_drawing(self):
        if self._drawing_enabled:
            self._drawing_enabled = False
            self.last_mode = self.mode
            self.mode = ImageCanvas.Drawing.NONE_MODE

    def enable_drawing(self):
        if not self._drawing_enabled:
            self._drawing_enabled = True
            self.mode = self.last_mode

    def toggle_mode(self, new_mode):
        if self._drawing_enabled:
            if self.img is not None:
                self._reset_polyshape_data()
                if self.mode != new_mode:
                    self.mode = new_mode
                    self.change_btns_mode(new_mode)
                else:
                    self.mode = ImageCanvas.Drawing.NONE_MODE
                    self.change_btns_mode()
                self._draw_current()

    def reset_mode(self):
        self.toggle_mode(ImageCanvas.Drawing.NONE_MODE)

    def toggle_flood_fill(self):
        self.toggle_mode(ImageCanvas.Drawing.FLOOD_FILL_MODE)

    def toggle_eraser(self):
        self.toggle_mode(ImageCanvas.Drawing.ERASER_MODE)

    def toggle_drawer(self):
        self.toggle_mode(ImageCanvas.Drawing.DRAW_MODE)

    def toggle_polyline(self):
        self.toggle_mode(ImageCanvas.Drawing.POLYLINE_MODE)

    def toggle_polygon(self):
        self.toggle_mode(ImageCanvas.Drawing.POLYGON_DRAW_MODE)

    def toggle_polygon_eraser(self):
        self.toggle_mode(ImageCanvas.Drawing.POLYGON_ERASER_MODE)

    def foreground_drawer(self):
        self.toggle_mode(ImageCanvas.Drawing.FOREGROUND_MODE)

    def background_drawer(self):
        self.toggle_mode(ImageCanvas.Drawing.BACKGROUND_MODE)

    def choose_seed(self):
        self.toggle_mode(ImageCanvas.Drawing.SEED_MODE)

    def toggle_box_drawer(self):
        self.toggle_mode(ImageCanvas.Drawing.BOX_MODE)

    def set_moving_masks(self, destination_layer, move):
        if self.img is not None:
            if move:
                self.destination_layer = destination_layer
                self.img.set_destination_layer(destination_layer)
            self.moving_mask = move

    def set_normalized_image_roi(self, image_roi: List[float]):
        self._normalized_image_roi = image_roi

        def set_roi(image_canvas):
            image_canvas.img.set_roi(self._normalized_image_roi)
            image_canvas._set_zoom()

        self.perform_action_on_image_and_redraw(lambda: set_roi(self))

    def undo_seeds(self):
        self.perform_action_on_image_and_redraw(lambda: self.img.undo_seed_mask())

    def undo_last_seed(self):
        self.perform_action_on_image_and_redraw(lambda: self.img.undo_last_seed_image())

    def undo_last_polyshape_operation(self):
        display_previous_polygon = False
        if self.polyshape_points_stack is None and self.prev_polyshape_finish_point_x is not None:
            self.polyshape_finish_point_x, self.polyshape_finish_point_y = (
                self.prev_polyshape_finish_point_x,
                self.prev_polyshape_finish_point_y,
            )
            self.polyshape_points_stack = self.polyshape_points_stack_prev
            display_previous_polygon = True
        if len(self.polyshape_points_stack) > 0:
            self.polyshape_points_stack.pop()
            if len(self.polyshape_points_stack) == 0:
                self.prev_polyshape_point_x, self.prev_polyshape_point_y, self.prev_polyshape_scale = None, None, None
                self.polyshape_finish_point_x, self.polyshape_finish_point_y = None, None
            else:
                (
                    self.prev_polyshape_point_x,
                    self.prev_polyshape_point_y,
                    self.prev_polyshape_scale,
                ) = self.polyshape_points_stack[-1]
            if self.mode == ImageCanvas.Drawing.POLYLINE_MODE:
                self.img.undo_polyline()
            elif self.mode in [ImageCanvas.Drawing.POLYGON_DRAW_MODE, ImageCanvas.Drawing.POLYGON_ERASER_MODE]:
                self.img.undo_polygon_line(display_previous_polygon)

    def undo_last_operation(self):
        if self.img is not None:
            last_operation = self.img.undo_last_operation_image()
            if last_operation in [
                MaskImage.Action.POLYGON_DRAW_MODE,
                MaskImage.Action.POLYGON_ERASER_MODE,
                MaskImage.Action.POLYLINE_MODE,
            ]:
                self.undo_last_polyshape_operation()
        self._draw_current()

    def perform_action_on_image_and_redraw(self, action):
        if self.img is not None:
            action()
        self._draw_current()

    def auto_mask_DL(self, channel):
        self.perform_action_on_image_and_redraw(lambda: self.img.generate_auto_mask_DL(channel))

    def undo_box(self):
        self.perform_action_on_image_and_redraw(lambda: self.img.undo_box_image())

    def delete_all_masks(self):
        self.perform_action_on_image_and_redraw(lambda: self.img.delete_all_masks())

    def grab_cut_with_rectangle(self):
        self.perform_action_on_image_and_redraw(lambda: self.img.generate_grab_cut_with_rectangle())

    def bright_auto_mask(self):
        self.perform_action_on_image_and_redraw(lambda: self.img.generate_bright_auto_mask())

    def lab_auto_mask(self):
        self.perform_action_on_image_and_redraw(lambda: self.img.generate_lab_automask())

    def clustering(self):
        self.perform_action_on_image_and_redraw(lambda: self.img.generate_clustering(self.settings.clusters_amount))

    def erosion(self):
        self.perform_action_on_image_and_redraw(lambda: self.img.apply_erosion(self.settings.erosion_iterations))

    def dilation(self):
        self.perform_action_on_image_and_redraw(lambda: self.img.apply_dilation(self.settings.dilation_iterations))

    def removing_objects(self):
        self.perform_action_on_image_and_redraw(
            lambda: self.img.apply_removing_objects(self.settings.removing_objects_size)
        )

    def closing_iterations_image_apply(self):
        self.perform_action_on_image_and_redraw(
            lambda: self.img.apply_closing_iterations(self.settings.closing_iterations_image)
        )

    def contrast(self):
        self.perform_action_on_image_and_redraw(lambda: self.img.set_contrast(self.contrast_value))

    def brightness(self):
        self.perform_action_on_image_and_redraw(lambda: self.img.set_brightness(self.brightness_value))

    def hue(self):
        self.perform_action_on_image_and_redraw(lambda: self.img.set_hue(self.hue_value))

    def mask_opacity(self):
        self.perform_action_on_image_and_redraw(lambda: self.img.set_mask_opacity(self.mask_opacity_value))

    def save_algo_stack_settings(self):
        if self.img is not None:
            algo_stack = list((str(x[0]), str(x[1])) for x in self.img.get_algo_stack())
            self.settings.algo_stack = json.dumps(algo_stack)

    def load_algo_stack_settings(self):
        if len(self.settings.algo_stack) == 0:
            return

        self.undo_all_masks()
        algo_stack = json.loads(self.settings.algo_stack)
        for algo, value in algo_stack:
            if algo == "Algorithm.AUTO_MASK_DL":
                if torchscript_model.is_loaded():
                    self.auto_mask_DL(value)
            elif algo == "Algorithm.LAB_AUTO_MASK":
                self.lab_auto_mask()
            elif algo == "Algorithm.CLUSTERING":
                self.settings.clusters_amount = int(value)
                self.clustering()
            elif algo == "Algorithm.BRIGHT_AUTO_MASK":
                self.bright_auto_mask()
            elif algo == "Algorithm.DILATION":
                self.settings.dilation_iterations = int(value)
                self.dilation()
            elif algo == "Algorithm.EROSION":
                self.settings.erosion_iterations = int(value)
                self.erosion()
            elif algo == "Algorithm.REMOVING_OBJECTS":
                self.settings.removing_objects_size = int(value)
                self.removing_objects()
            elif algo == "Algorithm.CLOSING_ITERATIONS":
                self.settings.closing_iterations_image = int(value)
                self.closing_iterations_image_apply()

    def small_brush(self):
        if self.mode == ImageCanvas.Drawing.POLYGON_DRAW_MODE:
            self.polygon_start_zoom -= 1
        else:
            self.brush_zoom -= 1
        self._draw_current()

    def big_brush(self):
        if self.mode == ImageCanvas.Drawing.POLYGON_DRAW_MODE:
            self.polygon_start_zoom += 1
        else:
            self.brush_zoom += 1
        self._draw_current()

    def undo_all_masks(self):
        self.perform_action_on_image_and_redraw(lambda: self.img.apply_undo_all_masks())

    def undo_mask(self):
        self.perform_action_on_image_and_redraw(lambda: self.img.undo_mask_image())

    def _reset_sliders(self, slider=None):
        def set_slider_silently(slider, value):
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)

        if self.sliders is not None:
            if Settings.Sliders.CLUSTERING in self.sliders and slider is not Settings.Sliders.CLUSTERING:
                self.settings.clusters_amount = DEFAULT_CLUSTERS_AMOUNT
                set_slider_silently(self.sliders[Settings.Sliders.CLUSTERING], self.settings.clusters_amount)
            if Settings.Sliders.REMOVING_OBJECTS in self.sliders and slider is not Settings.Sliders.REMOVING_OBJECTS:
                self.settings.removing_objects_size = DEFAULT_REMOVING_OBJECTS_SIZE
                set_slider_silently(
                    self.sliders[Settings.Sliders.REMOVING_OBJECTS], self.settings.removing_objects_size
                )
            if Settings.Sliders.CLOSING in self.sliders and slider is not Settings.Sliders.CLOSING:
                self.settings.closing_iterations_image = DEFAULT_CLOSING_ITERATIONS
                set_slider_silently(self.sliders[Settings.Sliders.CLOSING], self.settings.closing_iterations_image)
            if Settings.Sliders.DILATION in self.sliders and slider is not Settings.Sliders.DILATION:
                self.settings.dilation_iterations = DEFAULT_DILATION_ITERATIONS
                set_slider_silently(self.sliders[Settings.Sliders.DILATION], self.settings.dilation_iterations)
            if Settings.Sliders.EROSION in self.sliders and slider is not Settings.Sliders.EROSION:
                self.settings.erosion_iterations = DEFAULT_EROSION_ITERATIONS
                set_slider_silently(self.sliders[Settings.Sliders.EROSION], self.settings.erosion_iterations)

    def set_clusters_amount(self, slide):
        self._reset_sliders(Settings.Sliders.CLUSTERING)
        self.settings.clusters_amount = slide

    def set_removing_objects_size(self, slide):
        self._reset_sliders(Settings.Sliders.REMOVING_OBJECTS)
        self.settings.removing_objects_size = slide
        self.removing_objects()

    def set_closing_iterations_image(self, slide):
        self._reset_sliders(Settings.Sliders.CLOSING)
        self.settings.closing_iterations_image = slide
        self.closing_iterations_image_apply()

    def set_dilation_iterations(self, slide):
        self._reset_sliders(Settings.Sliders.DILATION)
        self.settings.dilation_iterations = slide
        self.dilation()

    def set_erosion_iterations(self, slide):
        self._reset_sliders(Settings.Sliders.EROSION)
        self.settings.erosion_iterations = slide
        self.erosion()

    def nullify_image_settings(self):
        if self.image_settings_sliders is not None:
            for count in range(len(self.image_settings_sliders)):
                self.image_settings_sliders[count].setValue(0)

    def set_contrast_level(self, slide):
        self.contrast_value = slide
        self.contrast()

    def set_brightness_level(self, slide):
        self.brightness_value = slide
        self.brightness()

    def set_hue_level(self, slide):
        self.hue_value = slide
        self.hue()

    def set_opacity_level(self, slide):
        self.mask_opacity_value = slide
        self.mask_opacity()

    @property
    def has_depth(self):
        return self.img.depth is not None
