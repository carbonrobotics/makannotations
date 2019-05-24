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

import time


class TimeElement:
    def __init__(self, name):
        self.when = time.time()
        self.name = name


class Timer:
    def __init__(self, name):
        self.name = name
        self.points = []

    def __enter__(self):
        self.mark("ENTER: " + self.name)
        return self

    def mark(self, what):
        self.points.append(TimeElement(what))

    def __exit__(self, t, v, tb):
        self.mark("EXIT: " + self.name)
        last = None
        for p in self.points:
            if last is None:
                print(p.name)
            else:
                print("{}: {} -> {}".format(self.name, "%.10f" % (p.when - last.when), p.name))
            last = p
