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

import os

import boto3
from botocore.exceptions import ClientError


def file_exists(path: str) -> bool:
    if path.startswith("s3://"):
        parts = path[5:].split("/")
        bucket = parts[0]
        path = "/".join(parts[1:])
        s3 = boto3.resource("s3")
        object = s3.Object(bucket, path)
        try:
            object.load()
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise
    else:
        return os.path.exists(path)


def read_file(path: str) -> bytes:
    if path.startswith("s3://"):
        parts = path[5:].split("/")
        bucket = parts[0]
        path = "/".join(parts[1:])
        s3 = boto3.resource("s3")
        object = s3.Object(bucket, path)
        try:
            return object.get()["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise
    else:
        if not os.path.exists(path):
            return None

        with open(path, "rb") as f:
            return f.read()


def write_file(path: str, data: bytes) -> None:
    if path.startswith("s3://"):
        parts = path[5:].split("/")
        bucket = parts[0]
        path = "/".join(parts[1:])
        s3 = boto3.resource("s3")
        object = s3.Object(bucket, path)
        object.put(Body=data)
    else:
        with open(path, "wb") as f:
            f.write(data)


def delete_file(path: str) -> None:
    if path.startswith("s3://"):
        parts = path[5:].split("/")
        bucket = parts[0]
        path = "/".join(parts[1:])
        s3 = boto3.resource("s3")
        object = s3.Object(bucket, path)
        object.delete()
    else:
        os.remove(path)