""" Template used for organizing potentials -- specifying active/inactive
splines, seeding with previous splines, etc.
"""

import numpy as np
import src.meam
from src.meam import MEAM

class Template:
    def __init__(self, pvec, spline_delimiters, active_splines=None):
        """
        Args:
            pvec (int): the parameter vector
            spline_delimiters (list[int]): indices separating spline params
        """

        self.pvec = pvec
        self.spline_delimiters = spline_delimiters

        self.set_active_splines(active_splines)

    def set_active_splines(self, active_splines):
        if active_splines is None:
            self.active_splines = np.ones(len(spline_delimiters) + 1)
        else:
            self.active_splines = np.array(active_splines)

        self.active_delimiters = []

        # self.active_delimiters is a list of [start, stop) tuples
        for i,is_active in enumerate(active_splines):
            if is_active:
                self.active_delimiters.append(
                    (self.spline_delimiters[i-1], self.spline_delimiters[i])
                    )

    @classmethod
    def from_file(cls, file_path):
        """
        Args:
            file_path (str): full path to LAMMPS-style potential file
        """
        pot = MEAM.from_file(file_path)

        x_pvec, y_pvec, delimiters = src.meam.splines_to_pvec(pot.splines)

        return cls()

    def get_active_pvec(self):
        """Returns that parameters that are being optimized"""

        active = []

        for start,stop in self.active_delimiters:
            active += [self.pvec[start:stop]]

        return np.hstack(active)

    def update_active_pvec(self, new_pvec):
        counter = 0
        for start,stop in self.active_delimiters:
            diff = stop - start

            self.pvec[start:stop] = new_pvec[counter:counter+diff]
            counter += diff

    def split_pvec(self):
        return np.array_split(self.pvec, self.spline_delimiters)

if __name__ == "__main__":
    pot = Template(
        np.arange(137),
        [15, 30, 45, 58, 71, 77, 83, 95, 107, 117, 127],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        )

    print("Full pvec:\n", pot.pvec)
    print("Delimiters:\n\t", pot.active_delimiters)
    print("Split pvec:")
    for i,spline in enumerate(pot.split_pvec()):
        print("\t", i, spline)
    print("Active pvec:\n\t", pot.get_active_pvec())
