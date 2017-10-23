"""Generates N random potentials and 'subtypes' potentials for use in testing"""
import numpy as np

from spline import Spline
from spline import ZeroSpline
from meam import MEAM
from .globalVars import a0, r0

num_knots = 11 # -> 10 intervals

# Potentials: zero_potential, constant_potential, rhs_extrap_potential,
# lhs_extrap_potential

################################################################################

"""Zero potential: gives 0.0 in all cases"""

splines = [ZeroSpline(np.arange(num_knots))]*12

zero_potential = MEAM(splines=splines, types=['H','He'])

################################################################################

"""Constant potential: this potential is designed to test pair/triplet 
contributions independently. Here is what each subtype should evaluate to:

    meam = 3*#pairs + #triplets
    nophi = 2*#pairs + #triplets
    norho = #pairs + #triplets
    norhophi = #triplets
    phionly = #pairs
    rhos = 2*#pairs
    rhophi = 3*#pairs"""

# all ones
phi_spline = Spline(np.linspace(0,a0, num=num_knots), np.ones(num_knots),
                    bc_type=((1,0),(1,0)), derivs=(0,0))

# all ones
rho_spline = Spline(np.linspace(0,a0, num=num_knots), np.ones(num_knots),
                  bc_type=((1,0),(1,0)), derivs=(0,0))

# u = ni, linear function
u_spline = Spline(np.linspace(0,a0, num=num_knots), np.linspace(0,a0, num=num_knots),
                    bc_type=((1,1),(1,1)), derivs=(1,1))

# all ones
f_spline = Spline(np.linspace(0,a0, num=num_knots), np.ones(num_knots),
                    bc_type=((1,0),(1,0)), derivs=(0,0))

# all ones
g_spline = Spline(np.linspace(0,a0, num=num_knots), np.ones(num_knots),
                    bc_type=((1,0),(1,0)), derivs=(0,0))

splines = [phi_spline]*3 + [rho_spline]*2 + [u_spline]*2 + [f_spline]*2 +\
                [g_spline]*3

constant_potential = MEAM(splines=splines, types=['H','He'])

################################################################################

"""Extrapolation potentials: identical to the constant potential, but with the
cutoff set shifted so that extrapolation is necessary. LHS extrapolation by 
shifting lower cutoff above r0, RHS extrapolation by shifting upper cutoff 
below r0. Extrapolated values will have a higher value."""

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

#rzm: testing? or build 1 random spline. Then change workertests.py

################################################################################
# TODO: create a special method that can take in a potential (or file) and test

'''
meams       = [None]*N # contributing functions: phi, u, rho, f, g
nophis      = [None]*N # contributing functions:      u, rho, f, g
phionlys    = [None]*N # contributing functions: phi
rhos        = [None]*N # contributing functions:      u, rho
norhos      = [None]*N # contributing functions: phi, u,      f, g
norhophis   = [None]*N # contributing functions:      u,      f, g
rhophis     = [None]*N # contributing functions: phi, u, rho

num_knots = 9
knots_x = np.linspace(0,a0, num=num_knots)
for n in range(N):
    # Generate splines with 10 knots, random y-coords of knots, equally
    # spaced x-coords ranging from 0 to a0, random d0 dN
    splines = []
    for i in range(12): # 2-component system has 12 total splines

        knots_y = np.random.random(num_knots)

        d0 = np.random.rand()
        dN = np.random.rand()

        temp = Spline(knots_x, knots_y, bc_type=((1,d0),(1,dN)),\
                derivs=(d0,dN))

        temp.cutoff = (knots_x[0],knots_x[len(knots_x)-1])
        splines.append(temp)

    tmp_splines = list(splines)

    # NOTE: order of splines is [phi,phi,phi,rho,rho,u,u,f,f,g,g,g]

    meams[n] = MEAM(splines=splines, types=['H','He'])

    splines = [tmp_splines[i] if (i>=3) else ZeroSpline(knots_x) for i in range(12)]
    nophis[n] = MEAM(splines = splines, types=['H','He'])

    splines = [tmp_splines[i] if (i<3) else ZeroSpline(knots_x) for i in range(12)]
    phionlys[n] = MEAM(splines = splines, types=['H','He'])

    splines = [tmp_splines[i] if ((i>=3) and (i<7)) else ZeroSpline(knots_x) for i in range(12)]
    rhos[n] = MEAM(splines = splines, types=['H','He'])

    splines = [tmp_splines[i] if ((i<3) or (i>4)) else ZeroSpline(knots_x) for i in range(12)]
    norhos[n] = MEAM(splines = splines, types=['H','He'])

    splines = [tmp_splines[i] if (i>4) else ZeroSpline(knots_x) for i in range(12)]
    norhophis[n] = MEAM(splines = splines, types=['H','He'])

    splines = [tmp_splines[i] if (i<7) else ZeroSpline(knots_x) for i in range(12)]
    rhophis[n] = MEAM(splines = splines, types=['H','He'])

allPotentials = {'meams':meams, 'nophis':nophis, 'phionlys':phionlys,\
        'rhos':rhos, 'norhos':norhos, 'norhophis':norhophis, 'rhophis':rhophis}

# TODO: may be worth making a function to get LAMMPS eval for a set of
# atoms/potentials

print("Created %d potentials (%d main, %d subtypes)" % (7*N, N, 6*N))'''
