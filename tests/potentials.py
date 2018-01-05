"""Generates N random potentials and 'subtypes' potentials for use in testing"""
import numpy as np
import meam
import sys

from spline import Spline
from spline import ZeroSpline
from meam import MEAM
from .globalVars import a0, r0

# Potentials: zero_potential, constant_potential, rhs_extrap_potential,
# lhs_extrap_potential

N                   = 0
zero_potential      = None
constant_potential  = None
rng_meams           = None
rng_nophis          = None
rng_phionlys        = None
rng_rhos            = None
rng_norhos          = None
rng_norhophis       = None
rng_rhophis         = None

################################################################################
def get_zero_potential():
    """Zero potential: gives 0.0 in all cases"""

    num_knots = 11

    global zero_potential

    if zero_potential:
        return zero_potential
    else:
        # splines = [ZeroSpline(np.arange(num_knots))]*12
        splines = np.zeros((12,num_knots,2))
        d0 = np.zeros(12)
        dN = np.zeros(12)

        indices = [12*i for i in range(1,12)]

        x = np.arange(12*num_knots, dtype=float)
        y = np.zeros(12*num_knots)
        y = np.append(y, d0); y = np.append(y, dN)

        y2 = y.copy()
        all_d0 = np.zeros(12)
        all_dN = np.zeros(12)

        # return knot_x_points, knot_y_points, y2, indices, all_d0, all_dN
        return x, y, y2, indices, all_d0, all_dN
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
        # # all ones
        # phi_spline = Spline(np.linspace(1,a0+1, num=num_knots), np.ones(num_knots),
        #                     bc_type=((1,0),(1,0)), derivs=(0,0))

        # # all ones
        # rho_spline = Spline(np.linspace(1,a0+1, num=num_knots), np.ones(num_knots),
        #                   bc_type=((1,0),(1,0)), derivs=(0,0))

        # # u = ni, linear function
        # u_spline = Spline(np.linspace(1,a0+1, num=num_knots), np.linspace(1,a0+1, num=num_knots),
        #                     bc_type=((1,1),(1,1)), derivs=(1,1))

        # # all ones
        # f_spline = Spline(np.linspace(1,a0+1, num=num_knots), np.ones(num_knots),
        #                     bc_type=((1,0),(1,0)), derivs=(0,0))

        # # all ones
        # g_spline = Spline(np.linspace(1,a0+1, num=num_knots), np.ones(num_knots),
        #                     bc_type=((1,0),(1,0)), derivs=(0,0))

        # splines = [phi_spline]*3 + [rho_spline]*2 + [u_spline]*2 + [f_spline]*2 +\
        #                 [g_spline]*3

        splines = np.ones((12,num_knots,2))

        # set u-splines to a linear function rather than all ones
        splines[5,:,0] = np.linspace(1, a0+1, num=num_knots)
        splines[5,:,1] = np.linspace(1, a0+1, num=num_knots)
        splines[6,:,0] = np.linspace(1, a0+1, num=num_knots)
        splines[6,:,1] = np.linspace(1, a0+1, num=num_knots)

        d0 = np.zeros(12)
        dN = np.zeros(12)

        constant_potential = MEAM(splines=splines, types=['H','He'], d0=d0,
                                  dN=dN)

        return constant_potential

################################################################################
def get_extrap_potentials():
    """Extrapolation potentials: identical to the constant potential, but with the
    cutoff set shifted so that extrapolation is necessary. LHS extrapolation by
    shifting lower cutoff above r0, RHS extrapolation by shifting upper cutoff
    below r0. Extrapolated values will have a higher value."""

    global rhs_extrap_potential
    global lhs_extrap_potential

    if rhs_extrap_potential and lhs_extrap_potential:
        return rhs_extrap_potential, lhs_extrap_potential
    else:
        # RHS extrapolation
        phi_spline = Spline(np.linspace(0,0.9*r0, num=num_knots), np.ones(num_knots),
                            bc_type=((1,0),(1,1)), derivs=(0,1))

        u_spline = Spline(np.linspace(0,0.9*r0, num=num_knots), np.linspace(0,0.9*r0, num=num_knots),
                            bc_type=((1,0),(1,1)), derivs=(0,1))

        rho_spline = Spline(np.linspace(0,0.9*r0, num=num_knots), np.ones(num_knots),
                          bc_type=((1,0),(1,1)), derivs=(0,1))

        f_spline = Spline(np.linspace(0,0.9*r0, num=num_knots), np.ones(num_knots),
                            bc_type=((1,0),(1,1)), derivs=(0,1))

        g_spline = Spline(np.linspace(0,0.9*r0, num=num_knots), np.ones(num_knots),
                            bc_type=((1,0),(1,1)), derivs=(0,1))

        splines = [phi_spline]*3 + [rho_spline]*2 + [u_spline]*2 + [f_spline]*2 +\
                        [g_spline]*3

        rhs_extrap_potential = MEAM(splines=splines, types=['H','He'])

        # LHS extrapolation
        rng = np.linspace(1.1*r0, 1.1*r0+a0, num=num_knots)

        phi_spline = Spline(rng, np.ones(num_knots),
                            bc_type=((1,0),(1,1)), derivs=(0,1))

        u_spline = Spline(rng, np.linspace(0,0.9*r0, num=num_knots),
                            bc_type=((1,0),(1,1)), derivs=(0,1))

        rho_spline = Spline(rng, np.ones(num_knots),
                          bc_type=((1,0),(1,1)), derivs=(0,1))

        f_spline = Spline(rng, np.ones(num_knots),
                            bc_type=((1,0),(1,1)), derivs=(0,1))

        g_spline = Spline(rng, np.ones(num_knots),
                            bc_type=((1,0),(1,1)), derivs=(0,1))

        splines = [phi_spline]*3 + [rho_spline]*2 + [u_spline]*2 + [f_spline]*2 +\
                        [g_spline]*3

        lhs_extrap_potential = MEAM(splines=splines, types=['H','He'])

        return rhs_extrap_potential, lhs_extrap_potential

################################################################################
# TODO: create a special method that can take in a potential (or file) and test

def get_random_pots(newN):

    global N
    global rng_meams
    global rng_nophis
    global rng_phionlys
    global rng_rhos
    global rng_norhos
    global rng_norhophis
    global rng_rhophis

    if N == newN:
        allpots = {'meams':rng_meams, 'nophis':rng_nophis,
                   'phionlys':rng_phionlys, 'rhos':rng_rhos,
                   'norhos':rng_norhos, 'norhophis':rng_norhophis,
                   'rhophis':rng_rhophis}
        return allpots
    else:
        N = newN
        rng_meams       = [None]*N
        rng_nophis      = [None]*N
        rng_phionlys    = [None]*N
        rng_rhos        = [None]*N
        rng_norhos      = [None]*N
        rng_norhophis   = [None]*N
        rng_rhophis     = [None]*N

        num_knots = 11

        knots_x = np.linspace(0, a0, num=num_knots)
        for n in range(N):
            # Generate splines with 10 knots, random y-coords of knots, equally
            # spaced x-coords ranging from 0 to a0, random d0 dN
            # splines = []

            splines = np.random.random((12,num_knots,2))*a0
            splines[:,:,0] = knots_x

            d0 = np.random.rand(12)
            dN = np.random.rand(12)

            # for i in range(12): # 2-component system has 12 total splines

                # knots_y = np.random.random(num_knots)

                # d0 = np.random.rand()
                # dN = np.random.rand()

                # temp = Spline(knots_x, knots_y, bc_type=((1,d0),(1,dN)),\
                #         derivs=(d0,dN))

                # temp.cutoff = (knots_x[0],knots_x[len(knots_x)-1])
                # splines.append(temp)

            p = MEAM(splines=splines, types=['H','He'], d0=d0, dN=dN)
            # p = MEAM('HHe.meam.spline')

            rng_meams[n]        = p
            rng_nophis[n]       = meam.nophi_subtype(p)
            rng_phionlys[n]     = meam.phionly_subtype(p)
            rng_rhos[n]         = meam.rho_subtype(p)
            rng_norhos[n]       = meam.norhophi_subtype(p)
            rng_norhophis[n]    = meam.norhophi_subtype(p)
            rng_rhophis[n]      = meam.rhophi_subtype(p)

        allpots = {'meams':rng_meams, 'nophis':rng_nophis,
                   'phionlys':rng_phionlys, 'rhos':rng_rhos,
                   'norhos':rng_norhos, 'norhophis':rng_norhophis,
                   'rhophis':rng_rhophis}

        return allpots

    #print("Created %d potentials (%d main, %d subtypes)" % (7*N, N, 6*N))

#allPotentials = {'meams':meams, 'nophis':nophis, 'phionlys':phionlys,\
#        'rhos':rhos, 'norhos':norhos, 'norhophis':norhophis, 'rhophis':rhophis}

# TODO: may be worth making a function to get LAMMPS eval for a set of
# atoms/potentials
