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

import io
import re
import subprocess

import cv2
import numpy as np

import boto3

_dir_re = re.compile("\s*PRE\s(.*)$")  # noqa
_file_re = re.compile("[\d-]*\s*[\d:]*\s*\d*\s*(.*)$")  # noqa


#
# Helpers
#


def regex_result(name, reg):
    m = reg.match(name)
    if m is not None:
        name = m.group(1)
        if len(name) < 1:
            return None
        if name[-1] == "/":
            name = name[:-1]
        if len(name) < 1 or name == ".":
            return None
        return name
    return None


def get_directory_name(dname):
    return regex_result(dname, _dir_re)


def get_file_name(fname):
    return regex_result(fname, _file_re)


def get_cmd_output(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE)
    outlines = proc.stdout.decode("utf-8").split("\n")
    return outlines


#
# API starts here
#


def get_directory_listing(path):
    """get the list of all directories under path.

       We can't use boto3 for this because it only supports Prefix search (ie you see all the files in all the
       directories below you). awscli does this properly but it has no external methods, so we shell out to it."""

    cmd = ["aws", "s3", "ls", "s3://" + path + "/"]
    outlines = get_cmd_output(cmd)
    list = []
    for l in outlines:
        dname = get_directory_name(l)
        if dname is not None:
            list.append(dname)
    return list


def get_file_listing(path):
    """get the list of all files under path.

       We can't use boto3 for this because it only supports Prefix search (ie you see all the files in all the
       directories below you). awscli does this properly but it has no external methods, so we shell out to it."""

    cmd = ["aws", "s3", "ls", "s3://" + path + "/"]
    outlines = get_cmd_output(cmd)
    list = []
    for l in outlines:
        fname = get_file_name(l)
        if fname is not None:
            list.append(fname)
    return list


def get_file_image_data(metadata, path, filename):
    """Returns the full data of the file requested under path."""
    parts = path.split("/")
    bucket = parts[0]
    path = "/".join(parts[1:] + [filename])
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket)
    objects = bucket.objects.filter(Prefix=path)
    objects = [o for o in objects]
    if len(objects) == 0:
        return None
    object = objects[0]
    data = object.get()["Body"].read()

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

    data = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.byte), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return data, None
