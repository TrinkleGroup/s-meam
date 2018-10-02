""" Template used for organizing potentials -- specifying active/inactive
splines, seeding with previous splines, etc.
"""

import numpy as np
import src.meam
from src.meam import MEAM

class Template:
    def __init__(self,
        pvec_len, spline_delimiters, spline_ranges, active_splines=[],
        load_file_path=None, seed=[]):
        """
        A tool for generating new potentials based off a given template. For
        example, it may be necessary to set certain splines to given values, but
        then to easily extract and change the portions that are variable.

        Args:
            pvec_len (int): the length of the parameter vector
            spline_delimiters (list): indices separating unique splines
            spline_ranges (list): list of [low, high] tuples for each spline
            active_splines (list): list of 0/1 where 1 means active
            input_file_name (str): path to LAMMPS-style potential to use as base
            seed (np.arr): starting format for pvec
        """

        self.load_file_path = load_file_path

        # Initialize properties
        if load_file_path:
            pot = MEAM.from_file(file_path)

            x_pvec, y_pvec, delimiters = src.meam.splines_to_pvec(pot.splines)
            self.pvec = y_pvec
            self.spline_delimiters = delimiters

            # TODO: ranges could be taken from LAMMPS file
        else:
            if len(seed) > 0:
                self.pvec = seed
            else:
                self.pvec = np.zeros(pvec_len)

            self.spline_delimiters = spline_delimiters

        self.spline_ranges = spline_ranges

        if len(active_splines) < 0:
            active_splines = np.ones(len(spline_delimiters) + 1)

        self.active_splines = active_splines
        self.active_mask = np.zeros(len(self.pvec), dtype=int)

        self.spline_delimiters.append(len(self.pvec))

        start = 0
        for i,active in enumerate(self.active_splines):
            stop = self.spline_delimiters[i]

            if active == 1:
                self.active_mask[start:stop] = 1

            start = stop

    def generate_random_instance(self, rng):
        """
        Instantiates a random instance of the template, using the given random
        number generator in the range of each spline
        """

        ind = np.zeros(len(self.pvec))

        start = 0
        for i in range(len(self.active_splines)):
            stop = self.spline_delimiters[i]

            low, high = self.spline_ranges[i]

            ind[start:stop] = rng(stop-start)*(high-low) + low

            start = stop

        return ind

    def insert_active_splines(self, new_pvec):
        tmp = self.pvec.copy()
        tmp[np.where(self.active_mask)] = new_pvec

        return tmp

    # def set_active_splines(self, active_splines):
    #     if active_splines is None:
    #         self.active_splines = np.ones(len(spline_delimiters) + 1)
    #     else:
    #         self.active_splines = np.array(active_splines)
    #
    #     self.active_delimiters = []
    #
    #     # self.active_delimiters is a list of [start, stop) tuples
    #     for i,is_active in enumerate(active_splines):
    #         if is_active:
    #             self.active_delimiters.append(
    #                 (self.spline_delimiters[i-1], self.spline_delimiters[i])
    #                 )
    #
    @classmethod
    def from_file(cls, file_path):
        """
        Args:
            file_path (str): full path to LAMMPS-style potential file
        """

        self.load_file_str = file_path
        return cls()

    def update_active_pvec(self, new_pvec):
        counter = 0
        for start,stop in self.active_delimiters:
            diff = stop - start

            self.pvec[start:stop] = new_pvec[counter:counter+diff]
            counter += diff

    def split_pvec(self):
        return np.array_split(self.pvec, self.spline_delimiters)

    def print_statistics(self):
        print("Loaded from:", self.load_file_path)
        print("pvec_len:", len(self.pvec))
        print("spline_delimiters:", self.spline_delimiters)
        print("spline_ranges:", self.spline_ranges)
        print("active_splines:", self.active_splines)

if __name__ == "__main__":
    pot = Template(
        137,
        spline_ranges= [(-1, 4), (-1, 4), (-1, 4), (-9, 3), (-9, 3), (-0.5, 1),
                        (-0.5, 1), (-2,3), (-2, 3), (-7,2), (-7,2), (-7,2)],
        spline_delimiters=[15, 30, 45, 58, 71, 77, 83, 95, 107, 117, 127],
        active_splines=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        seed=np.arange(137)
        )

    pot.print_statistics()
