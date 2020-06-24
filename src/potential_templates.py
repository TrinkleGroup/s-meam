""" Template used for organizing potentials -- specifying active/inactive
splines, seeding with previous splines, etc.
"""

import numpy as np
import src.meam
from src.meam import MEAM


class Template:
    def __init__(self, types,
                 pvec_len, u_ranges, active_mask=None, spline_indices=None,
                 spline_ranges=None, load_file_path=None, seed=None):
        """
        A tool for generating new potentials based off a given template. For
        example, it may be necessary to set certain splines to given values, but
        then to easily extract and change the portions that are variable.

        Args:
            types (list): list of element types
            pvec_len(int): size of parameter vector
            u_ranges (np.arr): 2 params for each USpline specifying knot ranges
            active_mask (np.arr): 1 where parameter is 'active'
            spline_indices (list): list of (start, stop) tuples for each spline
            spline_ranges (list): list of (low, high) tuples for each spline
            spline_tags(np.arr): identifies spline for each point in pvec
            scales (np.arr): ranges of spline knots
            load_file_path (str): path to LAMMPS-style potential to use as base
            seed (np.arr): starting format for pvec
        """

        if active_mask is not None:
            self.active_mask = active_mask
        else:
            self.active_mask = np.ones(pvec_len)

        # NOTE: active_mask is a binary mask for toggle_u_only_optimization()
        # NOTE: ^ that's stupid, I'm changing it back
        # self.active_mask = np.where(self.active_mask)[0]

        self.types = types
        self.ntypes = len(types)
        self.pvec_len = pvec_len
        self.u_ranges = u_ranges
        self.spline_ranges = spline_ranges
        self.spline_indices = spline_indices
        self.spline_tags = None
        self.rho_indices = None
        self.f_indices = None
        self.g_indices = None
        self.scales = None
        self.knot_positions = None
        self.x_indices = None
        self.cutoffs = None

        self.load_file_path = load_file_path

        # Initialize properties
        if load_file_path:
            pot = MEAM.from_file(load_file_path)

            x_pvec, y_pvec, delimiters = src.meam.splines_to_pvec(pot.splines)
            self.pvec = y_pvec

            # TODO: ranges could be taken from LAMMPS file
        elif seed is not None:
            self.pvec = seed.copy()
        else:
            self.pvec = self.generate_random_instance()

    def get_active_params(self):
        return self.pvec[np.where(self.active_mask)[0]]

    def generate_random_instance(self):
        """
        Instantiates a random instance of the template, using the given random
        number generator in the range of each spline
        """

        ind = np.zeros(self.pvec_len)

        spline_num = 0
        for ind_tup, rng_tup in zip(self.spline_indices, self.spline_ranges):

            start, stop = ind_tup
            low, high = rng_tup

            # TODO: hard-coded for binary
            # if spline_num < 5:
            #     seed = np.linspace(high, low, stop-start)
            #
            #     ind[start:stop] = seed + np.random.normal(
            #                                 size=(stop-start), scale=0.1)
            # else:
            #     ind[start:stop] = np.random.random(stop - start) * (high-low) + low

            ind[start:stop] = np.random.random(stop - start) * (high-low) + low

            spline_num += 1

        return ind

    def insert_active_splines(self, new_pvec):
        tmp = np.tile(self.pvec, (new_pvec.shape[0], 1))
        tmp[:, np.where(self.active_mask)[0]] = new_pvec

        return tmp

    def print_statistics(self):
        print("Loaded from:", self.load_file_path)
        print("pvec", self.pvec)
        print("active_mask", self.active_mask)
        print("spline_ranges:", self.spline_ranges)
        print("spline_indices:", self.spline_indices, flush=True)
