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

import fnmatch
import json
import hashlib
import os
from typing import List, Optional

from file_io import read_file, write_file

META_JSON = "meta.json"
DEFAULT_CERTIFICATION_VERSION = 1
DEFAULT_CERTIFICATION_SOURCE = "makannotations"
DEFAULT_MD5SUM = None


class ImageMetadata:
    def __init__(
        self,
        filepath: str,
        ppi: Optional[int] = None,
        npz_rgb_key: Optional[str] = None,
        npz_depth_key: Optional[str] = None,
        grid_size: Optional[List[int]] = None,
    ):
        self._filepath = filepath
        self._ppi = ppi
        self._npz_rgb_key = npz_rgb_key
        self._npz_depth_key = npz_depth_key
        self._grid_size = grid_size

    @property
    def filepath(self) -> str:
        return self._filepath

    @property
    def ppi(self) -> Optional[int]:
        return self._ppi

    @property
    def npz_rgb_key(self) -> Optional[str]:
        return self._npz_rgb_key

    @property
    def npz_depth_key(self) -> Optional[str]:
        return self._npz_depth_key

    @property
    def grid_size(self) -> Optional[List[int]]:
        return self._grid_size


class _PatternMetadata:
    def __init__(
        self,
        pattern: str,
        ppi: Optional[int] = None,
        npz_rgb_key: Optional[str] = None,
        npz_depth_key: Optional[str] = None,
        grid_size: Optional[List[int]] = None,
    ):
        self._pattern = pattern
        self._ppi = ppi
        self._npz_rgb_key = npz_rgb_key
        self._npz_depth_key = npz_depth_key
        self._grid_size = grid_size

    @property
    def pattern(self) -> str:
        return self._pattern

    @property
    def ppi(self) -> Optional[int]:
        return self._ppi

    @property
    def npz_rgb_key(self) -> Optional[str]:
        return self._npz_rgb_key

    @property
    def npz_depth_key(self) -> Optional[str]:
        return self._npz_depth_key

    @property
    def grid_size(self) -> Optional[List[int]]:
        return self._grid_size


class DirectoryMetadata:
    def __init__(self, dirpath: str, patterns: List[_PatternMetadata]):
        self._dirpath = dirpath
        self._patterns = patterns

    def get_image_metadata(self, filename: str) -> Optional[ImageMetadata]:
        if not self._patterns:
            # If no patterns are defined, return dummy metadata for all images
            if filename.endswith(".npz"):
                # Can't use .npz files without metadata
                return None
            return ImageMetadata(filepath=os.path.join(self._dirpath, filename))
        for pat in self._patterns:
            if fnmatch.fnmatch(filename, pat.pattern):
                if filename.endswith(".npz") and pat.npz_rgb_key is None and pat.npz_depth_key is None:
                    # Ignore .npz files without both RGB and depth information
                    continue
                return ImageMetadata(
                    filepath=os.path.join(self._dirpath, filename),
                    ppi=pat.ppi,
                    npz_rgb_key=pat.npz_rgb_key,
                    npz_depth_key=pat.npz_depth_key,
                    grid_size=pat.grid_size,
                )
        # If did not match any patterns, return None metadata indicating this image should not be listed
        return None

    @staticmethod
    def load(dirpath: str) -> "DirectoryMetadata":
        path = os.path.join(dirpath, META_JSON)
        patterns = []
        meta_json_bytes = read_file(path)
        if meta_json_bytes is not None:
            try:
                meta_json = json.loads(meta_json_bytes)
                for pat_json in meta_json:
                    if not pat_json.get("makannotations", True):
                        continue
                    pat = _PatternMetadata(
                        pattern=pat_json["pattern"],
                        ppi=pat_json.get("ppi"),
                        npz_rgb_key=pat_json.get("npz_rgb_key"),
                        npz_depth_key=pat_json.get("npz_depth_key"),
                        grid_size=pat_json.get("grid_size"),
                    )
                    patterns.append(pat)
            except Exception as e:
                print(f"Could not deserialize {path}: {e}")
        return DirectoryMetadata(dirpath=dirpath, patterns=patterns)


class CertificationData:
    def __init__(
        self,
        version=1,
        certified=False,
        username=None,
        source=DEFAULT_CERTIFICATION_SOURCE,
        timestamp=None,
        md5sum=DEFAULT_MD5SUM,
        hard_example=False,
    ):
        self._version = version
        self._certified = certified
        self._username = username
        self._source = source
        self._timestamp = timestamp
        self._md5sum = md5sum
        self._hard_example = hard_example

    @property
    def certified(self):
        return self._certified

    @property
    def md5sum(self):
        return self._md5sum

    @property
    def hard_example(self):
        return self._hard_example

    @property
    def source(self):
        return self._source

    @property
    def md5sum(self):
        return self._md5sum

    def modify(self, certified, mask_modified, hard_example):
        return self._certified != certified or mask_modified or self._hard_example != hard_example

    def get_source(self, mask_modified):
        return DEFAULT_CERTIFICATION_SOURCE if mask_modified else self._source

    @staticmethod
    def calculate_md5sum(mask_filename):
        mask_bytes = read_file(mask_filename)
        if mask_bytes is not None:
            m = hashlib.md5()
            m.update(mask_bytes)
            return m.hexdigest()
        else:
            return DEFAULT_MD5SUM

    def get_md5sum(self, mask_modified, mask_filename):
        if not mask_modified:
            return self._md5sum

        return CertificationData.calculate_md5sum(mask_filename)

    def to_json(self):
        return {
            "version": 1,
            "certified": self._certified,
            "username": self._username,
            "source": self._source,
            "timestamp": self._timestamp,
            "md5sum": self._md5sum,
            "hard_example": self._hard_example,
        }

    def write(self, filepath):
        write_file(filepath, json.dumps(self.to_json(), indent=4).encode())

    @staticmethod
    def from_json(json):
        return CertificationData(
            version=json["version"],
            certified=json["certified"],
            username=json["username"],
            source=json["source"],
            timestamp=json["timestamp"],
            md5sum=json["md5sum"],
            hard_example=json.get("hard_example", False),
        )

    @staticmethod
    def load(filepath):
        config_bytes = read_file(filepath)
        if config_bytes is not None:
            try:
                config = json.loads(config_bytes)
                return CertificationData.from_json(config)
            except Exception as e:
                print("certification file load failure:", e)
        return CertificationData()

    @staticmethod
    def make_certification_filename(path, image_filename, layer):
        return os.path.join(path, os.path.splitext(image_filename)[0] + ".mask_{}.json".format(layer))
