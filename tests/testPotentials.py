"""Generates N random potentials and 'subtypes' potentials for use in testing"""
import numpy as np

from spline import Spline
from spline import ZeroSpline
from meam import MEAM
from .testVars import a0

N = 0
zero_potential = None
constant_potential = None
rng_meams = None
rng_nophis = None
rng_phionlys = None
rng_rhos = None
rng_norhos = None
rng_norhophis = None
rng_rhophis = None


################################################################################
def get_zero_potential():
    """Zero potential: gives 0.0 in all cases"""

    num_knots = 11

    global zero_potential

    if zero_potential:
        return zero_potential
    else:
        splines = [ZeroSpline(np.arange(num_knots))] * 12

        zero_potential = MEAM(splines=splines, types=['H', 'He'])

        return zero_potential


################################################################################
def get_constant_potential():
    """Constant potential: this potential is designed to test pair/triplet
    contributions independently. Here is what each subtype should evaluate to:

        meam = 3*#pairs + #triplets
        nophi = 2*#pairs + #triplets
        norho = #pairs + #triplets
        norhophi = #triplets
        phionly = #pairs
        rhos = 2*#pairs
        rhophi = 3*#pairs"""

    num_knots = 11

    global constant_potential

    if constant_potential:
        return constant_potential
    else:
        # all ones
        phi_spline = Spline(np.linspace(1, a0 + 1, num=num_knots),
                            np.ones(num_knots),
                            bc_type=((1, 0), (1, 0)), end_derivs=(0, 0))

        # all ones
        rho_spline = Spline(np.linspace(1, a0 + 1, num=num_knots),
                            np.ones(num_knots),
                            bc_type=((1, 0), (1, 0)), end_derivs=(0, 0))

        # u = ni, linear function
        u_spline = Spline(np.linspace(1, a0 + 1, num=num_knots),
                          np.linspace(1, a0 + 1, num=num_knots),
                          bc_type=((1, 1), (1, 1)), end_derivs=(1, 1))

        # all ones
        f_spline = Spline(np.linspace(1, a0 + 1, num=num_knots),
                          np.ones(num_knots),
                          bc_type=((1, 0), (1, 0)), end_derivs=(0, 0))

        # all ones
        g_spline = Spline(np.linspace(1, a0 + 1, num=num_knots),
                          np.ones(num_knots),
                          bc_type=((1, 0), (1, 0)), end_derivs=(0, 0))

        splines = [phi_spline] * 3 + [rho_spline] * 2 + [u_spline] * 2 + [
            f_spline] * 2 + [g_spline] * 3

        constant_potential = MEAM(splines=splines, types=['H', 'He'])

        return constant_potential


################################################################################
def get_random_pots(new_n):
    global N
    global rng_meams
    global rng_nophis
    global rng_phionlys
    global rng_rhos
    global rng_norhos
    global rng_norhophis
    global rng_rhophis

    if N == new_n:
        allpots = {'meams': rng_meams, 'nophis': rng_nophis,
                   'phionlys': rng_phionlys, 'rhos': rng_rhos,
                   'norhos': rng_norhos, 'norhophis': rng_norhophis,
                   'rhophis': rng_rhophis}
        return allpots
    else:
        N = new_n
        rng_meams = [None] * N
        rng_nophis = [None] * N
        rng_phionlys = [None] * N
        rng_rhos = [None] * N
        rng_norhos = [None] * N
        rng_norhophis = [None] * N
        rng_rhophis = [None] * N

        num_knots = 11
        knots_x = np.linspace(0, a0, num=num_knots)
        for n in range(N):
            # Generate splines with 10 knots, random y-coords of knots, equally
            # spaced x-coords ranging from 0 to a0, random d0 dN
            splines = []
            for i in range(12):  # 2-component system has 12 total splines

                knots_y = np.random.random(num_knots)

                d0 = np.random.rand()
                dN = np.random.rand()

                # if i!=7:
                #     knots_y = np.ones(num_knots)
                #     knots_y = np.arange(num_knots)/10.
                #     d0 = dN = 0

                if i > 8:
                    knots_x = np.linspace(-1, 1, num=num_knots)

                temp = Spline(knots_x, knots_y, bc_type=((1, d0), (1, dN)), \
                              end_derivs=(d0, dN))

                temp.cutoff = (knots_x[0], knots_x[len(knots_x) - 1])
                splines.append(temp)

            p = MEAM(splines=splines, types=['H', 'He'])
            # p = MEAM.from_file("../data/pot_files/HHe.meam.spline")

            # import meam
            # x_pvec, y_pvec, indices = meam.splines_to_pvec(p.splines)
            # splines = meam.splines_from_pvec(x_pvec, y_pvec, indices)
            # p = MEAM(splines=splines, types=p.types)

            rng_meams[n] = p
            rng_nophis[n] = p.nophi_subtype()
            rng_phionlys[n] = p.phionly_subtype()
            rng_rhos[n] = p.rho_subtype()
            rng_norhos[n] = p.norhophi_subtype()
            rng_norhophis[n] = p.norhophi_subtype()
            rng_rhophis[n] = p.rhophi_subtype()

        allpots = {'meams': rng_meams, 'nophis': rng_nophis,
                   'phionlys': rng_phionlys, 'rhos': rng_rhos,
                   'norhos': rng_norhos, 'norhophis': rng_norhophis,
                   'rhophis': rng_rhophis}
        return allpots

    # print("Created %d potentials (%d main, %d subtypes)" % (7*N, N, 6*N))

# allPotentials = {'meams':meams, 'nophis':nophis, 'phionlys':phionlys,\
#        'rhos':rhos, 'norhos':norhos, 'norhophis':norhophis, 'rhophis':rhophis}

# TODO: may be worth making a function to get LAMMPS eval for a set of
# atoms/potentials
