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

import cv2
import numpy as np

FG_BG_NONE_VAL = 0
FG_BG_FOREGROUND_VAL = 1
FG_BG_BACKGROUND_VAL = 2
FG_BG_BOX_VAL = 3

HSV_DARK_GREEN = [50, 129, 120]
HSV_LIGHT_GREEN = [100, 255, 255]
HSV_MAX_VALUE = [180, 255, 255]


def grab_cut_algo(image, grab_cut_mask, bounding_rect, refine_flag, resize_scale=3):
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    if refine_flag:
        refine_image = cv2.resize(image, None, fx=1 / resize_scale, fy=1 / resize_scale)
        refine_grab_cut_mask = cv2.resize(grab_cut_mask, None, fx=1 / resize_scale, fy=1 / resize_scale)
        cv2.grabCut(refine_image, refine_grab_cut_mask, None, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK)
        grab_cut_mask = cv2.resize(refine_grab_cut_mask, grab_cut_mask.shape[1::-1])
    else:
        cv2.grabCut(image, grab_cut_mask, bounding_rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
        x, y, w, h = bounding_rect
        grab_cut_mask = grab_cut_mask[y : y + h, x : x + w]

    return np.where((grab_cut_mask == 2) | (grab_cut_mask == 0), False, True).astype("bool")


def k_means_clustering(original_image, clusters_amount, seed_mask, resize_scale=2):
    # Make the image smaller to reduce k-means run time.
    image = cv2.resize(original_image, None, fx=1 / resize_scale, fy=1 / resize_scale)

    green_channel = image[:, :, 1].reshape(-1)
    a_channel = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 1].reshape(-1)
    vectorized = np.dstack((green_channel, a_channel))
    vectorized = np.float32(vectorized.reshape((-1, 2)))

    # Run k-means on vectorized image, get labels for each pixel.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(
        vectorized, clusters_amount, None, criteria=criteria, attempts=2, flags=cv2.KMEANS_PP_CENTERS
    )

    # Convert labels to 2d array.
    clustered_image = labels.reshape(image.shape[:2])

    # Get set of labels that are present within the seed (make seed the same size as image).
    seed_mask = cv2.resize(seed_mask, None, fx=1 / resize_scale, fy=1 / resize_scale)
    seed_clusters = list(set(clustered_image[np.where(seed_mask == 1)]))

    # Mask is those pixels that are in the clusters which are present in seed_clusters.
    mask = np.isin(clustered_image, seed_clusters).astype("uint8")
    return cv2.resize(mask, original_image.shape[1::-1]).astype("bool")


def image_lab_automask(rgb_img):
    img_lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2Lab)
    a_LAB_channel = img_lab[:, :, 1]
    _, thresh = cv2.threshold(a_LAB_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh == 0


def image_bright_auto_mask(rgb_img):
    img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([HSV_DARK_GREEN])
    upper_green = np.array([HSV_LIGHT_GREEN])
    img = cv2.inRange(img, lower_green, upper_green)

    kernel = np.ones((3, 3), "uint8")
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    return img == 255
