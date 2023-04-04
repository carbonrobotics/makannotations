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

    cmd = ["aws", "s3", "ls", path + "/"]
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

    cmd = ["aws", "s3", "ls", path + "/"]
    outlines = get_cmd_output(cmd)
    list = []
    for l in outlines:
        fname = get_file_name(l)
        if fname is not None:
            list.append(fname)
    return list
