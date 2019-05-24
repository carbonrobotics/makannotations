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

try:
    has_torch = True
    import torch
    import torch.nn.functional as F
except Exception:
    has_torch = False


MODEL = None


def load(path):
    if not has_torch:
        return
    global MODEL

    MODEL = torch.jit.load(path, map_location="cpu")
    MODEL.eval()


def is_loaded():
    if not has_torch:
        return False
    global MODEL

    return MODEL is not None


def image_auto_mask(rgb_img, channel, threshold=0.5):
    if not has_torch:
        return None
    global MODEL

    # Convert to PyTorch tensor
    t = torch.tensor([rgb_img]).permute(0, 3, 1, 2).float()
    t = F.interpolate(t, (720, 1280), mode="bilinear", align_corners=False)

    # Standard RGB adjustments
    t /= 255.0
    t[:, 0, :, :] -= 0.485
    t[:, 1, :, :] -= 0.456
    t[:, 2, :, :] -= 0.406
    t[:, 0, :, :] /= 0.229
    t[:, 1, :, :] /= 0.224
    t[:, 2, :, :] /= 0.225

    # Run inference
    with torch.no_grad():
        mask = MODEL(t)

    # Resize mask back to original image size
    mask = F.interpolate(mask, (rgb_img.shape[0], rgb_img.shape[1]), mode="bilinear", align_corners=False)

    # Mask is assumed to be on channel=1
    return (mask[0][channel] > threshold).numpy()
