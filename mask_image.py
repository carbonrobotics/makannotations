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

import datetime
import io
import os

import cv2
import matplotlib
import numpy as np
from aenum import AutoNumberEnum
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image, ImageEnhance
from skimage import morphology
from skimage.segmentation import flood_fill
from typing import List, Tuple

import torchscript_model
from file_io import read_file
from foreground_background_detection import (
    grab_cut_algo,
    image_bright_auto_mask,
    image_lab_automask,
    k_means_clustering,
)
from metadata import CertificationData, DEFAULT_CERTIFICATION_VERSION, DEFAULT_CERTIFICATION_SOURCE
from utils import decode_color, make_texture

matplotlib.use("agg")

GRID_ROWS = 4
GRID_COLUMNS = 3

MASK_UNDO_STACK_LEN = 100
SAVE_ALWAYS = True
DEFAULT_MASK_COLOR = [255, 0, 0]
POLYGON_LINE_ERASER_COLOR = [0, 0, 0]


def load_image(metadata, path, filename):
    data = read_file(path, filename)

    if filename.endswith(".npz"):
        image_meta = metadata.get_image_metadata(filename)
        assert image_meta is not None
        assert image_meta.npz_rgb_key is not None or image_meta.npz_depth_key is not None

        np_file = io.BytesIO(data)
        image_dict = np.load(np_file)

        if image_meta.npz_rgb_key is not None:
            image = image_dict[image_meta.npz_rgb_key]
        else:
            image = np.zeros(image_dict[image_meta.npz_depth_key].shape[0:2] + (3,), dtype=np.uint8)

        if image_meta.npz_depth_key is not None:
            depth = image_dict[image_meta.npz_depth_key]
        else:
            depth = None

        return image, depth

    return cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.byte), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), None


def create_mask(img, mina, maxa, minb, maxb, minc, maxc):
    """ generic rgb vs hsv"""
    return np.logical_and(
        np.logical_and(np.greater_equal(img[:, :, 0], mina), np.less_equal(img[:, :, 0], maxa)),
        np.logical_and(np.greater_equal(img[:, :, 1], minb), np.less_equal(img[:, :, 1], maxb)),
        np.logical_and(np.greater_equal(img[:, :, 2], minc), np.less_equal(img[:, :, 2], maxc)),
    )


def create_rgb_mask(img, minr=0, maxr=255, ming=0, maxg=255, minb=0, maxb=255):
    return create_mask(img, mina=minr, maxa=maxr, minb=ming, maxb=maxg, minc=minb, maxc=maxb)


def create_hsv_mask(img, minh=0, maxh=255, mins=0, maxs=255, minv=0, maxv=255):
    return create_mask(img, mina=minh, maxa=maxh, minb=mins, maxb=maxs, minc=minv, maxc=maxv)


def apply_multiple_masks(colors_masks, outimg, mask_opacity):
    for color, mask in colors_masks:
        apply_mask(outimg, mask, mask_opacity, color)


def apply_mask(outimg, mask, mask_opacity, color):
    outimg_before = outimg.copy()
    # save contours to be able to draw an outline
    contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # make texture
    rgb_color, texture_id = decode_color(color)
    texture = make_texture((mask.shape[0], mask.shape[1]), texture_id)
    mask = mask & texture
    # apply texture to the image
    outimg[mask] = rgb_color
    # draw contours on top of texture
    cv2.drawContours(outimg, contours, -1, rgb_color, thickness=1)
    # combine original image with weighted image
    cv2.addWeighted(outimg, mask_opacity, outimg_before, 1 - mask_opacity, 0, outimg)


def image_apply_mask(
    outimg,
    img,
    mask,
    seed_mask,
    boxes_stack,
    polygon_lines,
    polygon_line_color,
    mask_opacity,
    mask_color,
    x,
    xlen,
    y,
    ylen,
):
    if x is not None and xlen is not None and y is not None and ylen is not None:
        outimg[y : y + ylen, x : x + xlen] = img[y : y + ylen, x : x + xlen]
        cropped_outimg = outimg[y : y + ylen, x : x + xlen]
        apply_mask(cropped_outimg, mask[y : y + ylen, x : x + xlen], mask_opacity, mask_color)
    else:
        outimg[:, :] = img
        apply_mask(outimg, mask, mask_opacity, mask_color)
    outimg[seed_mask == 1] = [80, 180, 80]
    box_color = [0, 0, 0]
    for box in boxes_stack:
        # deliberately swap x and y
        y_begin, x_begin, y_end, x_end = box
        outimg[x_begin : x_begin + 3, y_begin:y_end] = box_color
        outimg[x_end : x_end + 3, y_begin:y_end] = box_color
        outimg[x_begin:x_end, y_begin : y_begin + 3] = box_color
        outimg[x_begin:x_end, y_end : y_end + 3] = box_color
    for line in polygon_lines:
        cur_line_start, cur_line_end = line
        cv2.line(outimg, tuple(cur_line_start), tuple(cur_line_end), polygon_line_color, 1)


def mask_image_from_image(outimg, mask, x, xlen, y, ylen):
    if x is not None and xlen is not None and y is not None and ylen is not None:
        outimg[y : y + ylen, x : x + xlen] = 0
        outimg[y : y + ylen, x : x + xlen][mask[y : y + ylen, x : x + xlen]] = [0, 0, 255]
    else:
        outimg[:, :] = 0
        outimg[mask] = [0, 0, 255]


def image_apply_inch_grid(outimg, inch_grid):
    outimg[inch_grid, :] = 0


def channel_values(chans):
    return chans.min(), chans.max()


def index_of_elements_last_occurrence(stack, element):
    return len(stack) - 1 - stack[::-1].index(element)


def load_mask(label_file, mask):
    if not os.path.isfile(label_file):
        return np.zeros_like(mask)
    label_mask = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
    label_mask = label_mask.astype("bool")
    np.logical_or(label_mask, mask, mask)

    return label_mask


def save_mask(label_filename, mask):
    if not mask.any():
        if os.path.isfile(label_filename):
            os.remove(label_filename)
        return
    mask = np.where(mask == True, 255, 0).astype("uint8")  # noqa
    cv2.imwrite(label_filename, mask)


class MaskImage:
    class Algorithm(AutoNumberEnum):
        GRAB_CUT_WITH_RECTANGLE = ()
        LAB_AUTO_MASK = ()
        AUTO_MASK_DL = ()
        BRIGHT_AUTO_MASK = ()
        CLUSTERING = ()
        REMOVING_OBJECTS = ()
        CLOSING_ITERATIONS = ()
        DILATION = ()
        EROSION = ()
        MASK_MOVE = ()

    class Action(AutoNumberEnum):
        MASK_MODE = ()
        BOX_MODE = ()
        SEED_MODE = ()
        POLYGON_DRAW_MODE = ()
        POLYGON_ERASER_MODE = ()
        POLYLINE_MODE = ()
        MASK_MOVE = ()

    ALGORITHMS_STACK = [
        Algorithm.LAB_AUTO_MASK,
        Algorithm.CLUSTERING,
        Algorithm.BRIGHT_AUTO_MASK,
        Algorithm.DILATION,
        Algorithm.EROSION,
        Algorithm.REMOVING_OBJECTS,
        Algorithm.CLOSING_ITERATIONS,
        Algorithm.AUTO_MASK_DL,
    ]

    class Settings(AutoNumberEnum):
        BRIGHTNESS = ()
        CONTRAST = ()

    BORDER_SIZE = 10

    @staticmethod
    def make_label_filename(path, image_filename, layer):
        return os.path.join(path, os.path.splitext(image_filename)[0] + ".mask_{}.png".format(layer))

    @staticmethod
    def from_file(
        filename,
        loader,
        label_path,
        mask_image,
        mask_only,
        show_depth,
        layer,
        show_inch_grid,
        ppi_value,
        destination_layer,
        mask_opacity,
        image_has_mask_callback,
        update_layer_has_mask,
        update_other_masks_information_callback,
        contrast,
        brightness,
        hue,
        show_all_layers,
        colors_masks,
        current_color,
        normalized_roi,
    ):
        img, depth = loader(filename)
        if img is None:
            return None
        return MaskImage.from_image(
            filename,
            label_path,
            img,
            depth,
            mask_image,
            mask_only,
            show_depth,
            layer,
            show_inch_grid,
            ppi_value,
            destination_layer,
            mask_opacity,
            image_has_mask_callback,
            update_layer_has_mask,
            update_other_masks_information_callback,
            contrast,
            brightness,
            hue,
            show_all_layers,
            colors_masks,
            current_color,
            normalized_roi,
        )

    @staticmethod
    def from_image(
        filename,
        label_path,
        img,
        depth,
        mask_image,
        mask_only,
        show_depth,
        layer,
        show_inch_grid,
        ppi_value,
        destination_layer,
        mask_opacity,
        image_has_mask_callback,
        update_layer_has_mask,
        update_other_layers_masks_callback,
        contrast,
        brightness,
        hue,
        show_all_layers,
        colors_masks,
        mask_color,
        normalized_roi,
    ):
        if img is not None:
            img = MaskImage(
                filename,
                img,
                depth,
                label_path,
                layer=layer,
                image_has_mask_callback=image_has_mask_callback,
                update_layer_has_mask=update_layer_has_mask,
                update_other_layers_masks_callback=update_other_layers_masks_callback,
                mask_image=mask_image,
                mask_only=mask_only,
                show_depth=show_depth,
                show_inch_grid=show_inch_grid,
                ppi_value=ppi_value,
                destination_layer=destination_layer,
                mask_opacity=mask_opacity,
                contrast=contrast,
                brightness=brightness,
                hue=hue,
                show_all_layers=show_all_layers,
                other_layers_masks=colors_masks,
                mask_color=mask_color,
                normalized_roi=normalized_roi,
            )
            label_filename = MaskImage.make_label_filename(label_path, filename, layer)
            img.load_image_mask(label_filename)
            return img
        return None

    def __init__(
        self,
        image_filename,
        img,
        depth,
        label_path,
        layer,
        image_has_mask_callback,
        update_layer_has_mask,
        update_other_layers_masks_callback,
        mask_image=False,
        mask_only=False,
        show_depth=False,
        show_inch_grid=False,
        ppi_value=None,
        destination_layer=None,
        mask_opacity=0,
        contrast=1,
        brightness=1,
        hue=1,
        show_all_layers=False,
        other_layers_masks=None,
        mask_color=DEFAULT_MASK_COLOR,
        normalized_roi=[0, 0, 1, 1],
    ):
        # Store original refernce image
        self.image_filename = image_filename
        self.origin_img = img
        self.depth = depth
        self.label_path = label_path
        self.mask_stack = []
        self.boxes_stack = []
        self.seeds_stack = []
        self.undo_stack = []
        self.seed_mask = np.zeros(img.shape[:2], dtype="uint8")
        self.mask_stack_top = 0
        self._initial_mask = np.zeros(img.shape[:2], dtype=np.bool)
        # Second element means algorithm
        self.mask_stack.append((self._initial_mask, None, None))
        self.visible_area = None

        # Settings
        self.contrast = MaskImage.calculate_contrast(contrast)
        self.brightness = MaskImage.calculate_brightness(brightness)
        self.hue = hue
        self.mask_opacity = MaskImage.calculate_mask_opacity(mask_opacity)

        # Polygon
        self.polygon_lines = []
        self.polygon_lines_prev = []
        self.polygon_lines_origin = []
        self.polygon_lines_origin_prev = []
        self.polygon_line_color = mask_color

        # Layering
        self.layer = layer
        self.destination_layer = destination_layer
        self.destination_layer_file = None
        self.last_destination_mask = None
        self.hidden_shapes = []

        # Images for display
        self._display = img
        self._outimg = np.zeros_like(img)
        self._img = img.copy()
        self._all_layers_mask = None
        self._roi_start = None
        self._roi_end = None
        self.set_roi(normalized_roi)

        # Controls
        self.mask_image = mask_image
        self.mask_only = mask_only
        self.show_depth = show_depth
        self.scale = 1.0
        self.hole = None
        self.last_hole_radius = None

        self.show_inch_grid = show_inch_grid
        self.ppi_value = ppi_value
        self.inch_grid = None

        self._mask_certified = False
        self._hard_example = False

        self.image_has_mask_callback = image_has_mask_callback
        self.update_layer_has_mask = update_layer_has_mask
        self._manual_mask_update = False

        self.show_all_layers = show_all_layers
        self.other_layers_colors_masks = other_layers_masks
        self.mask_color = mask_color

        self.update_other_layers_masks_callback = update_other_layers_masks_callback
        self._borders = []

        self.apply_settings()

    def set_mask_certified(self, certified):
        self._mask_certified = certified

    def set_hard_example(self, hard_example):
        self._hard_example = hard_example

    def save_certification_file(self):
        certification_filepath = CertificationData.make_certification_filename(
            self.label_path, self.image_filename, self.layer
        )
        mask_filepath = MaskImage.make_label_filename(self.label_path, self.image_filename, self.layer)
        previous_certification_data = CertificationData.load(certification_filepath)
        if previous_certification_data is None or previous_certification_data.modify(
            self._mask_certified, self._is_mask_modified(), self._hard_example
        ):
            new_certification_data = CertificationData(
                version=DEFAULT_CERTIFICATION_VERSION,
                certified=self._mask_certified,
                username=os.getenv("USER"),
                source=previous_certification_data.get_source(self._is_mask_modified()),
                timestamp=datetime.datetime.utcnow().isoformat(timespec="seconds"),
                md5sum=previous_certification_data.get_md5sum(self._is_mask_modified(), mask_filepath),
                hard_example=self._hard_example,
            )
            new_certification_data.write(certification_filepath)

    def _is_mask_modified(self):
        return not np.array_equal(self._initial_mask, self.mask())

    def load_image_mask(self, label_file):
        self._initial_mask = load_mask(label_file, self.mask())
        self._update_mask()

    def save_image_mask(self):
        label_filename = MaskImage.make_label_filename(self.label_path, self.image_filename, self.layer)
        if self._is_mask_modified():
            save_mask(label_filename, self.mask())

    def set_destination_layer(self, destination_layer):
        self.destination_layer = destination_layer

    def create_destination_layer_file(self):
        self.destination_layer_file = MaskImage.make_label_filename(
            self.label_path, self.image_filename, self.destination_layer
        )

    def move_mask(self):
        if MaskImage.Action.MASK_MOVE in self.undo_stack:
            self.undo_stack.pop(index_of_elements_last_occurrence(self.undo_stack, MaskImage.Action.MASK_MODE))
            self.undo_stack.pop(index_of_elements_last_occurrence(self.undo_stack, MaskImage.Action.MASK_MOVE))

        current_mask = self.mask().copy()
        mask = np.zeros_like(current_mask)

        x, y, x_end, y_end = self.boxes_stack[-1]
        mask[y:y_end, x:x_end] = current_mask[y:y_end, x:x_end]
        current_mask[y:y_end, x:x_end] = np.zeros((y_end - y, x_end - x)).astype(np.bool)

        self.create_destination_layer_file()
        md5sum = CertificationData.calculate_md5sum(self.destination_layer_file)
        self.last_destination_mask = load_mask(self.destination_layer_file, mask)
        save_mask(self.destination_layer_file, mask)
        new_md5sum = CertificationData.calculate_md5sum(self.destination_layer_file)
        if md5sum != new_md5sum:
            self.update_destination_layer_certification_data(new_md5sum)
            self.update_layer_has_mask(layer=self.destination_layer, has_mask=(new_md5sum is not None))

        self.boxes_stack = []
        self._set_mask(current_mask)

        self._manual_mask_update = True
        self.update_other_layers_masks_callback()
        self._update_mask()

    def undo_mask_move(self):
        if self.last_destination_mask is not None and self.destination_layer_file is not None:
            md5sum = CertificationData.calculate_md5sum(self.destination_layer_file)
            save_mask(self.destination_layer_file, self.last_destination_mask)
            new_md5sum = CertificationData.calculate_md5sum(self.destination_layer_file)
            if md5sum != new_md5sum:
                self.update_destination_layer_certification_data(new_md5sum)
                self.update_layer_has_mask(layer=self.destination_layer, has_mask=(new_md5sum is not None))
            self.undo_stack.pop()
            self.last_destination_mask = None
            self._manual_mask_update = True
            self.update_other_layers_masks_callback()

    def update_destination_layer_certification_data(self, md5sum):
        destination_layer_certification_filepath = CertificationData.make_certification_filename(
            self.label_path, self.image_filename, self.destination_layer
        )
        destination_layer_certification_data = CertificationData.load(destination_layer_certification_filepath)
        new_certification_data = CertificationData(
            version=DEFAULT_CERTIFICATION_VERSION,
            certified=destination_layer_certification_data.certified,
            username=os.getenv("USER"),
            source=DEFAULT_CERTIFICATION_SOURCE,
            timestamp=datetime.datetime.utcnow().isoformat(timespec="seconds"),
            md5sum=md5sum,
            hard_example=destination_layer_certification_data.hard_example,
        )
        new_certification_data.write(destination_layer_certification_filepath)

    def get_scale(self):
        return self.scale

    def get_algo_stack(self):
        return [
            (algo, value)
            for _, algo, value in self.mask_stack[: self.mask_stack_top + 1]
            if algo in MaskImage.ALGORITHMS_STACK
        ]

    def _set_mask(self, mask, algorithm=None, value=None):
        if len(self.mask_stack) >= MASK_UNDO_STACK_LEN:
            self.mask_stack = self.mask_stack[1:]
            self.mask_stack_top -= 1  # it is possible for this to go negative but it works out
        self.mask_stack = self.mask_stack[: self.mask_stack_top + 1]
        self.mask_stack.append((mask, algorithm, value))
        self.undo_stack.append(self.Action.MASK_MODE)
        self.mask_stack_top += 1
        self._manual_mask_update = True

    def mask(self, algorithm=None):
        if algorithm is not None and self.mask_stack[self.mask_stack_top][1] == algorithm:
            self.undo_mask_image()
            self._manual_mask_update = True
        return self.mask_stack[self.mask_stack_top][0]

    def generate_grab_cut_with_rectangle(self):
        if len(self.boxes_stack) > 0:
            x, y, x_end, y_end = self.boxes_stack[-1]
            w, h = x_end - x, y_end - y
            # Chopping image and assigning some of it to background.
            bg_window_size = 50
            x_image_begin = np.maximum(y - bg_window_size, 0)
            x_image_end = np.minimum(y_end + bg_window_size, self._img.shape[0])
            y_image_begin = np.maximum(x - bg_window_size, 0)
            y_image_end = np.minimum(x_end + bg_window_size, self._img.shape[1])
            box = (x - y_image_begin, y - x_image_begin, w, h)

            local_image = self._img[x_image_begin:x_image_end, y_image_begin:y_image_end].copy()
            result_mask = self.mask().copy()
            local_grab_cut_mask = np.zeros(local_image.shape[:2], dtype="uint8")
            result_mask[y : y + h, x : x + w] = grab_cut_algo(local_image, local_grab_cut_mask, box, False)

            self._set_mask(result_mask)
            self._update_mask()

    def generate_bright_auto_mask(self):
        self._set_mask(image_bright_auto_mask(self._img), MaskImage.Algorithm.BRIGHT_AUTO_MASK)
        self._update_mask()

    def generate_lab_automask(self):
        mask = self.mask(None).copy()
        if len(self.boxes_stack) == 0:
            self.generate_lab_automask_algo(mask, None)
        else:
            for box in self.boxes_stack:
                self.generate_lab_automask_algo(mask, box)
        self._set_mask(mask, MaskImage.Algorithm.LAB_AUTO_MASK)
        self._update_mask()

    def generate_lab_automask_algo(self, mask, box):
        if box is None:
            x, y, w, h = 0, 0, self._img.shape[1], self._img.shape[0]
        else:
            x, y, x_end, y_end = box
            w, h = x_end - x, y_end - y

        img = self._img[y : y + h, x : x + w].copy()
        mask[y : y + h, x : x + w] = image_lab_automask(img)

    def generate_auto_mask_DL(self, channel):
        self._set_mask(
            torchscript_model.image_auto_mask(self._img, channel), MaskImage.Algorithm.AUTO_MASK_DL, value=channel
        )
        self._update_mask()

    def generate_clustering(self, clusters_amount):
        if len(self.seeds_stack) > 0:
            mask = self.mask().copy()
            if len(self.boxes_stack) == 0:
                self.generate_clustering_algorithm(clusters_amount, mask, None)
            else:
                for box in self.boxes_stack:
                    self.generate_clustering_algorithm(clusters_amount, mask, box)
            self._set_mask(mask, MaskImage.Algorithm.CLUSTERING, clusters_amount)
            self._update_mask()

    def generate_clustering_algorithm(self, clusters_amount, mask, box):
        if box is None:
            x, y, w, h = 0, 0, self._img.shape[1], self._img.shape[0]
        else:
            x, y, x_end, y_end = box
            w, h = x_end - x, y_end - y

        image = self._img[y : y + h, x : x + w].copy()
        seed_mask = self.seed_mask[y : y + h, x : x + w].copy()
        mask[y : y + h, x : x + w] = k_means_clustering(image, clusters_amount, seed_mask)

    def apply_filter(self, algorithm_mode, action, value):
        mask = self.mask(algorithm_mode).copy().astype("uint8")
        if len(self.boxes_stack) == 0:
            self.generate_filter(None, mask, action)
        else:
            for box in self.boxes_stack:
                self.generate_filter(box, mask, action)

        self._set_mask(mask > 0, algorithm_mode, value)
        self._update_mask()

    def generate_filter(self, box, mask, action):
        if box is None:
            x, y, w, h = 0, 0, self._img.shape[1], self._img.shape[0]
        else:
            x, y, x_end, y_end = box
            w, h = x_end - x, y_end - y

        local_mask = mask[y : y + h, x : x + w].copy()
        local_mask = action(local_mask)
        mask[y : y + h, x : x + w] = local_mask

    def apply_dilation(self, dilation_iterations):
        def dilation(mask):
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            result_mask = cv2.dilate(mask, kernel=se, iterations=dilation_iterations)
            # Rotating because dilation works only from top and left sides of the image.
            result_mask = np.rot90(result_mask, 2)
            result_mask = cv2.dilate(result_mask, kernel=se, iterations=dilation_iterations)
            result_mask = np.rot90(result_mask, 2)

            return result_mask

        self.apply_filter(MaskImage.Algorithm.DILATION, dilation, dilation_iterations)

    def apply_erosion(self, erosion_iterations):
        def erosion(mask):
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            result_mask = cv2.erode(mask, kernel=se, iterations=erosion_iterations)
            # Rotating because erosion works only from top and left sides of the image.
            result_mask = np.rot90(result_mask, 2)
            result_mask = cv2.erode(result_mask, kernel=se, iterations=erosion_iterations)
            result_mask = np.rot90(result_mask, 2)

            return result_mask

        self.apply_filter(MaskImage.Algorithm.EROSION, erosion, erosion_iterations)

    def apply_removing_objects(self, removing_objects_size):
        def removing_objects(mask):
            mask = mask.astype("bool")
            result_mask = morphology.remove_small_objects(mask, removing_objects_size, connectivity=2).astype("uint8")

            return result_mask

        self.apply_filter(MaskImage.Algorithm.REMOVING_OBJECTS, removing_objects, removing_objects_size)

    def apply_closing_iterations(self, closing_iterations):
        def closing_iterations_operation(mask):
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            result_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=se, iterations=closing_iterations)
            # Rotating because closing works only from top and left sides of the image.
            result_mask = np.rot90(result_mask, 2)
            result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel=se, iterations=closing_iterations)
            result_mask = np.rot90(result_mask, 2)

            return result_mask

        self.apply_filter(MaskImage.Algorithm.CLOSING_ITERATIONS, closing_iterations_operation, closing_iterations)

    def apply_settings(self):
        img = self.origin_img.copy()

        img = Image.fromarray(img)
        img = self.adjust_contrast(img)
        img = self.adjust_brightness(img)
        img = np.asarray(img)

        img = self.adjust_hue(img)

        self._img = img
        self.create_all_layers_mask()
        self._update_mask()

    @staticmethod
    def adjust_settings(img, function, factor):
        enhancer = function(img)
        img = enhancer.enhance(factor)
        return img

    def adjust_contrast(self, img):
        return self.adjust_settings(img, ImageEnhance.Contrast, self.contrast)

    def adjust_brightness(self, img):
        return self.adjust_settings(img, ImageEnhance.Brightness, self.brightness)

    def adjust_hue(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_channel = img_hsv[:, :, 0]
        if self.hue < 0:
            h_channel = np.maximum(h_channel + self.hue, 0)
        else:
            h_channel = np.minimum(h_channel + self.hue, 255)
        img_hsv[:, :, 0] = h_channel
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        return img

    @staticmethod
    def calculate_brightness(brightness, scale=100):
        return 1 + brightness / scale

    def set_brightness(self, brightness):
        self.brightness = MaskImage.calculate_brightness(brightness)
        self.apply_settings()

    @staticmethod
    def calculate_contrast(contrast, scale=100):
        return 1 + contrast / scale

    def set_contrast(self, contrast):
        self.contrast = MaskImage.calculate_contrast(contrast)
        self.apply_settings()

    def set_hue(self, hue):
        self.hue = hue
        self.apply_settings()

    @staticmethod
    def calculate_mask_opacity(opacity, scale=100):
        return 0.5 + opacity / scale

    def set_mask_opacity(self, opacity):
        self.mask_opacity = MaskImage.calculate_mask_opacity(opacity)
        self.create_all_layers_mask()
        self._update_mask()

    def display_width(self):
        return self._display.shape[1]

    def display_height(self):
        return self._display.shape[0]

    def display_channels(self):
        if len(self._img.shape) > 2:
            return self._img.shape[2]
        return 1

    def get_display_hsv(self):
        return cv2.cvtColor(self._display, cv2.COLOR_RGB2HSV_FULL)

    def get_display_lab(self):
        return cv2.cvtColor(self._display, cv2.COLOR_RGB2Lab)

    def get_display_rgb(self):
        return self._display

    def set_scale(self, scale):
        self.scale = scale
        self._update_mask()

    def add_glowing_borders(self, img):
        img = img.copy()
        for point_1, point_2 in self._borders:
            cv2.line(img, point_1, point_2, (255, 69, 0), thickness=MaskImage.BORDER_SIZE)
        return img

    def _update_display(self):
        x_start, y_start, x_end, y_end = self.visible_area
        if self.mask_only or self.mask_image:
            img = self._outimg[y_start:y_end, x_start:x_end, :]
        else:
            img = self._img[y_start:y_end, x_start:x_end, :]

        img = self.add_glowing_borders(img)
        self._display = cv2.resize(img, dsize=None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        return self._display.astype("uint8")

    def _get_contour_depth(self, contour=32):
        assert 256 % contour == 0, f"Use contours evenly dividing 256, provided: {contour}"
        scale = int(256 / contour)
        return (cv2.cvtColor(self.depth, cv2.COLOR_GRAY2RGB) % contour) * scale

    def _update_mask(self, x=None, xlen=None, y=None, ylen=None):
        if self._manual_mask_update:
            self.image_has_mask_callback(self.mask().any())
            self._manual_mask_update = False

        img = None
        if self.mask_only:
            mask_image_from_image(self._outimg, self.mask(), x, xlen, y, ylen)
        elif self.show_depth:
            img = self._get_contour_depth()
        elif self.mask_image:
            if self.show_all_layers and self._all_layers_mask is not None:
                img = self._all_layers_mask
            else:
                img = self._img

        if img is not None:
            image_apply_mask(
                self._outimg,
                img,
                self.mask(),
                self.seed_mask,
                self.boxes_stack,
                self.polygon_lines,
                self.polygon_line_color,
                self.mask_opacity,
                self.mask_color,
                x,
                xlen,
                y,
                ylen,
            )

        if self.show_inch_grid and self.ppi_value:
            if self.inch_grid is None:
                self.find_inch_grid()
            image_apply_inch_grid(self._outimg, self.inch_grid)

    def apply_mask(self, value):
        self.mask_image = value
        self._update_mask()

    def only_mask(self, value):
        self.mask_only = value
        self._update_mask()

    def set_show_depth(self, value):
        self.show_depth = value
        self._update_mask()

    def find_inch_grid(self):
        adjusted_ppi = int(self.ppi_value / 4) * 4
        self.inch_grid = np.zeros(self._img.shape[:2], dtype=np.bool)
        self.inch_grid[:, :: adjusted_ppi] = True
        self.inch_grid[:: adjusted_ppi] = True
        # add quarter inch grid
        self.inch_grid[:: 3, :: int(adjusted_ppi / 4)] = True
        self.inch_grid[:: int(adjusted_ppi / 4), :: 3] = True

    def apply_inch_grid(self, show_grid):
        self.show_inch_grid = show_grid
        self._update_mask()

    def create_all_layers_mask(self):
        self._all_layers_mask = None
        if self.show_all_layers and self.other_layers_colors_masks is not None:
            self._all_layers_mask = self._img.copy()
            apply_multiple_masks(self.other_layers_colors_masks, self._all_layers_mask, self.mask_opacity)

    def set_show_all_layers(self, show_all_layers):
        self.show_all_layers = show_all_layers
        self.create_all_layers_mask()
        self._update_mask()

    def set_colors_masks(self, colors_masks):
        self.other_layers_colors_masks = colors_masks

    def set_mask_color(self, mask_color):
        self.mask_color = mask_color

    def convert_to_limits(self, x, y):
        x = self._roi_start[1] + x
        y = self._roi_start[0] + y
        return x, y

    def add_borders(self):
        self._borders = []
        x_start, y_start, x_end, y_end = self.visible_area
        if x_start != self._roi_start[1]:
            self._borders.append(((0, 0), (0, y_end - y_start)))
        if y_start != self._roi_start[0]:
            self._borders.append(((0, 0), (x_end - x_start, 0)))
        if x_end != self._roi_end[1]:
            self._borders.append(
                (
                    (x_end - x_start - MaskImage.BORDER_SIZE // 2, 0),
                    (x_end - x_start - MaskImage.BORDER_SIZE // 2, y_end - y_start),
                )
            )
        if y_end != self._roi_end[0]:
            self._borders.append(
                (
                    (0, y_end - y_start - MaskImage.BORDER_SIZE // 2),
                    (x_end - x_start, y_end - y_start - MaskImage.BORDER_SIZE // 2),
                )
            )

    def chopped_display(self, x, y, width, height):
        y_start = self._roi_start[0] + y
        x_start = self._roi_start[1] + x
        y_end = min(self._roi_end[0], y_start + height)
        x_end = min(self._roi_end[1], x_start + width)
        self.visible_area = (x_start, y_start, x_end, y_end)
        self.add_borders()
        self._update_display()

        return self._display

    @property
    def roi_width(self):
        return self._roi_end[1] - self._roi_start[1]

    @property
    def roi_height(self):
        return self._roi_end[0] - self._roi_start[0]

    def set_roi(self, roi: List[float]):
        start_width, start_height, end_width, end_height = roi

        self._roi_start = [int(start_height * self._img.shape[0]), int(start_width * self._img.shape[1])]
        self._roi_end = [int(end_height * self._img.shape[0]), int(end_width * self._img.shape[1])]

    def make_hole(self, radius):
        if self.hole is not None and self.last_hole_radius == radius:
            return
        yy, xx = np.mgrid[: radius * 2, : radius * 2]
        circle = (xx - radius) ** 2 + (yy - radius) ** 2
        hole = circle > (radius * radius)
        self.hole = hole.astype(np.bool)
        self.last_hole_radius = radius

    def _draw_xy(self, x, y, radius):
        radius = round(radius / self.scale)
        x = round(x / self.scale)
        y = round(y / self.scale)
        x -= radius
        y -= radius

        x, y = self.convert_to_limits(x, y)

        # We may have clipped off the edge
        holex_start = 0
        holey_start = 0
        ylen = 2 * radius
        xlen = 2 * radius
        if x < 0:
            holex_start = abs(x)
            xlen -= abs(x)
            x = 0
        if y < 0:
            holey_start = abs(y)
            ylen -= abs(y)
            y = 0

        if x + xlen > self.mask().shape[1]:
            xlen = self.mask().shape[1] - x
        if y + ylen > self.mask().shape[0]:
            ylen = self.mask().shape[0] - y

        return x, y, xlen, ylen, holex_start, holey_start, radius

    def mask_draw_erase(
        self, x, y, radius, mask, np_logical_or_and, np_logical_not_or_none=None, copy_last_mask=False,
    ):
        x, y, xlen, ylen, holex_start, holey_start, radius = self._draw_xy(x, y, radius)
        if mask[y : y + ylen, x : x + xlen].size == 0:
            # Drawing outside of viewing area
            return

        self.make_hole(radius)
        hole_part = self.hole[holey_start : holey_start + ylen, holex_start : holex_start + xlen]
        if np_logical_not_or_none is not None:
            hole_part = np_logical_not_or_none(hole_part)

        # This can happen if you click off and then back on again
        if hole_part.size == 0:
            return

        result = np_logical_or_and(mask[y : y + ylen, x : x + xlen], hole_part)

        mask[y : y + ylen, x : x + xlen] = result
        self._set_mask(mask if not copy_last_mask else self.mask())
        self._update_mask(x=x, xlen=xlen, y=y, ylen=ylen)

    def mask_erase(self, x, y, radius):
        self.mask_draw_erase(x, y, radius, self.mask().copy(), np.logical_and, None)

    def mask_draw(self, x, y, radius):
        self.mask_draw_erase(x, y, radius, self.mask().copy(), np.logical_or, np.logical_not)

    def line_roi(self, start_point, end_point, radius=5):
        start_x, start_y = start_point
        end_x, end_y = end_point

        x_min = max(min(start_x, end_x) - radius, 0)
        x_max = min(max(start_x, end_x) + radius, self._img.shape[1])

        y_min = max(min(start_y, end_y) - radius, 0)
        y_max = min(max(start_y, end_y) + radius, self._img.shape[0])

        return x_min, x_max - x_min, y_min, y_max - y_min

    def draw_erase_line(self, prev_point, cur_point, radius, np_ones_zeros, np_logical_or_and, color, set_mask=False):
        x_start, y_start = prev_point
        prev_point = self.convert_to_limits(x_start, y_start)
        cur_x, cur_y = cur_point
        cur_point = self.convert_to_limits(cur_x, cur_y)
        x, xlen, y, ylen = self.line_roi(prev_point, cur_point, radius)

        mask = self.mask()
        if set_mask:
            mask = mask.copy()

        tmpmask = np_ones_zeros(mask.shape, dtype=np.uint8)
        cv2.line(tmpmask, prev_point, cur_point, color, 2 * radius)
        np_logical_or_and(tmpmask, mask, mask)

        if set_mask:
            self._set_mask(mask)

        self._manual_mask_update = True
        self._update_mask(x=x, xlen=xlen, y=y, ylen=ylen)

    def draw_line(self, prev_point, cur_point, radius, set_mask=False):
        self.draw_erase_line(prev_point, cur_point, radius, np.zeros, np.logical_or, 255, set_mask)

    def erase_line(self, prev_point, cur_point, radius):
        self.draw_erase_line(prev_point, cur_point, radius, np.ones, np.logical_and, 0)

    def flood_fill(self, x, y):
        x, y = self.convert_to_limits(x, y)
        tmpmask = np.zeros_like(self.mask().copy()).astype("uint8")
        tmpmask[self.mask()] = 255
        mask = tmpmask.copy()

        mask = flood_fill(mask, (y, x), 255)
        self._set_mask(mask == 255)
        self._update_mask()

    def make_polygon_mask(self, points):
        fig = plt.figure(0, figsize=self.mask().shape[::-1], dpi=1)
        plt.axis("off")
        plt.axis("image")
        plt.axis([0, self.mask().shape[1], self.mask().shape[0], 0])
        ax = plt.gca()
        ax.add_artist(Polygon(points, color="black", fill=True))
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        canvas_bytes = fig.canvas.tostring_rgb()
        canvas_shape = fig.canvas.get_width_height()[::-1]
        plt.close(fig)

        data = np.frombuffer(canvas_bytes, dtype=np.uint8)
        data = data.reshape(canvas_shape + (3,))
        # Mask even slightly non-white pixels to support having very thin leaves
        return data[..., 0] < 255

    def fill_polygon(self, mode):
        points = []
        for line in self.polygon_lines_origin:
            start, end = line
            points.append(end)

        poly_mask = self.make_polygon_mask(points)
        mask = self.mask().copy()
        if mode == MaskImage.Action.POLYGON_DRAW_MODE:
            np.logical_or(poly_mask, mask, mask)
        elif mode == MaskImage.Action.POLYGON_ERASER_MODE:
            np.logical_and(np.logical_not(poly_mask), mask, mask)
        self._set_mask(mask)

        self.polygon_lines_origin_prev = self.polygon_lines_origin
        self.polygon_lines_prev = self.polygon_lines
        self.polygon_lines_origin = []
        self.polygon_lines = []
        self._update_mask()

    def reset_polygon_data(self):
        self.polygon_lines = []
        self.polygon_lines_prev = []
        self.polygon_lines_origin = []
        self.polygon_lines_origin_prev = []

        self.clean_undo_stack_polyshape()
        self._update_mask()

    def clean_undo_stack_polyshape(self):
        x = list(
            filter(
                lambda a: a
                not in [
                    MaskImage.Action.POLYGON_DRAW_MODE,
                    MaskImage.Action.POLYGON_ERASER_MODE,
                    MaskImage.Action.POLYLINE_MODE,
                ],
                self.undo_stack,
            )
        )
        self.undo_stack = x

    def scale_polyshape_line(self, start, end, prev_scale):
        x_start, y_start = start
        x_end, y_end = end
        x_start = round(x_start / prev_scale)
        x_end = round(x_end / self.scale)
        y_start = round(y_start / prev_scale)
        y_end = round(y_end / self.scale)

        return x_start, y_start, x_end, y_end

    def draw_polyline(self, start, end, prev_scale, thickness):
        x_start, y_start, x_end, y_end = self.scale_polyshape_line(start, end, prev_scale)

        self.undo_stack.append(MaskImage.Action.POLYLINE_MODE)
        self.draw_line((x_start, y_start), (x_end, y_end), thickness, set_mask=True)
        x, xlen, y, ylen = self.line_roi((x_start, y_start), (x_end, y_end))

        self._update_mask(x=x, xlen=xlen, y=y, ylen=ylen)

    def undo_polyline(self):
        self.undo_stack.pop(index_of_elements_last_occurrence(self.undo_stack, MaskImage.Action.POLYLINE_MODE))

    def _scale_line(self, start, end):
        x_start, y_start = start
        x_end, y_end = end
        x_start = round(x_start / self.scale)
        x_end = round(x_end / self.scale)
        y_start = round(y_start / self.scale)
        y_end = round(y_end / self.scale)

        return x_start, y_start, x_end, y_end

    def polygon_line(self, start, end, prev_scale, mode):
        x_start, y_start, x_end, y_end = self._scale_line(start, end)
        x_start, y_start = self.convert_to_limits(x_start, y_start)
        x_end, y_end = self.convert_to_limits(x_end, y_end)
        self.polygon_lines_origin.append(
            [[x_start, y_start], [x_end, y_end],]
        )

        x_start, y_start, x_end, y_end = self.scale_polyshape_line(start, end, prev_scale)
        x_start, y_start = self.convert_to_limits(x_start, y_start)
        x_end, y_end = self.convert_to_limits(x_end, y_end)
        self.polygon_lines.append([[x_start, y_start], [x_end, y_end]])
        if mode == MaskImage.Action.POLYGON_DRAW_MODE:
            self.polygon_line_color = self.mask_color
        else:
            self.polygon_line_color = POLYGON_LINE_ERASER_COLOR

        self.undo_stack.append(mode)
        self._update_mask()

    def undo_polygon_line(self, display_previous_polygon=False):
        if display_previous_polygon is True:
            self.polygon_lines = self.polygon_lines_prev
            self.polygon_lines_origin = self.polygon_lines_origin_prev
        if len(self.polygon_lines) > 0:
            self.polygon_lines.pop()
            self.polygon_lines_origin.pop()
        self.undo_stack.pop()
        self._update_mask()

    def box_draw(self, box, mode):
        x_start, y_start, x_end, y_end = box

        x_start = round(x_start / self.scale)
        x_end = round(x_end / self.scale)
        y_start = round(y_start / self.scale)
        y_end = round(y_end / self.scale)

        x_start, y_start = self.convert_to_limits(x_start, y_start)
        x_end, y_end = self.convert_to_limits(x_end, y_end)

        x_end = max(min(x_end, self._img.shape[1]), 0)
        y_end = max(min(y_end, self._img.shape[0]), 0)

        x_start, x_end = min(x_start, x_end), max(x_start, x_end)
        y_start, y_end = min(y_start, y_end), max(y_start, y_end)

        self.boxes_stack.append((x_start, y_start, x_end, y_end))

        if mode == self.Action.MASK_MOVE:
            self.move_mask()
        self.undo_stack.append(mode)

        self._update_mask()

    def undo_last_seed_image(self):
        if len(self.seeds_stack) > 0:
            x, y, xlen, ylen = self.seeds_stack[-1]
            self.seeds_stack.pop(-1)
            self.seed_mask[y : y + ylen, x : x + xlen] = 0

            self.undo_stack.pop(index_of_elements_last_occurrence(self.undo_stack, self.Action.SEED_MODE))
            self._update_mask()

    def undo_box_image(self):
        if len(self.boxes_stack) > 0:
            self.boxes_stack.pop(-1)

            self.undo_stack.pop(index_of_elements_last_occurrence(self.undo_stack, self.Action.BOX_MODE))
            self._update_mask()

    def delete_all_masks(self):
        mask = np.zeros(self._img.shape[:2], dtype="bool")
        self._set_mask(mask)
        self._update_mask()

    def seed_draw(self, x, y, radius):
        x, y, xlen, ylen, holex_start, holey_start, radius = self._draw_xy(x, y, radius)
        self.make_hole(radius)

        hole_part = self.hole[holey_start : holey_start + ylen, holex_start : holex_start + xlen]
        self.seed_mask[y : y + ylen, x : x + xlen][hole_part == 0] = 1

        self.seeds_stack.append((x, y, xlen, ylen))
        self.undo_stack.append(self.Action.SEED_MODE)

        self._update_mask()

    def undo_last_operation_image(self):
        if len(self.undo_stack) > 0:
            last_operation = self.undo_stack[-1]
            if last_operation == self.Action.MASK_MODE:
                self.undo_mask_image()
                if len(self.undo_stack) > 0:
                    if self.undo_stack[-1] in [
                        MaskImage.Action.POLYGON_DRAW_MODE,
                        MaskImage.Action.POLYGON_ERASER_MODE,
                        MaskImage.Action.POLYLINE_MODE,
                    ]:
                        last_operation = self.undo_stack[-1]
            elif last_operation == self.Action.BOX_MODE:
                self.undo_box_image()
            elif last_operation == self.Action.SEED_MODE:
                self.undo_last_seed_image()
            elif last_operation == self.Action.MASK_MOVE:
                self.undo_mask_move()
                self.undo_mask_image()

            self._update_mask()
            return last_operation

    def undo_seed_mask(self):
        self.seed_mask[:, :] = 0
        self._update_mask()

    def apply_undo_all_masks(self):
        self.mask_stack_top = 0

    def undo_mask_image(self):
        if self.mask_stack_top > 0:
            self.mask_stack_top -= 1
            self.undo_stack.pop(index_of_elements_last_occurrence(self.undo_stack, self.Action.MASK_MODE))
            self._manual_mask_update = True
            self._update_mask()
