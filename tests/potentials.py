"""Generates N random potentials and 'subtypes' potentials for use in testing"""
import numpy as np

from spline import Spline
from spline import ZeroSpline
from meam import MEAM
from .globalVars import a0

N = 10

# Random potential cutoff in range [1,10]; atomic distances 
# Default atomic spacing is set as a0/2
# Default vacuum is a0*2

meams       = [None]*N # contributing functions: phi, u, rho, f, g
nophis      = [None]*N # contributing functions:      u, rho, f, g
phionlys    = [None]*N # contributing functions: phi
rhos        = [None]*N # contributing functions:      u, rho
norhos      = [None]*N # contributing functions: phi, u,      f, g
norhophis   = [None]*N # contributing functions:      u,      f, g
rhophis     = [None]*N # contributing functions: phi, u, rho

num_knots = 10
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

    splines = [tmp_splines[i] if (i>=3) else ZeroSpline() for i in range(12)]
    nophis[n] = MEAM(splines = splines, types=['H','He'])

    splines = [tmp_splines[i] if (i<3) else ZeroSpline() for i in range(12)]
    phionlys[n] = MEAM(splines = splines, types=['H','He'])

    splines = [tmp_splines[i] if ((i>=3) and (i<7)) else ZeroSpline() for i in range(12)]
    rhos[n] = MEAM(splines = splines, types=['H','He'])

    splines = [tmp_splines[i] if ((i<3) or (i>4)) else ZeroSpline() for i in range(12)]
    norhos[n] = MEAM(splines = splines, types=['H','He'])

    splines = [tmp_splines[i] if (i>4) else ZeroSpline() for i in range(12)]
    norhophis[n] = MEAM(splines = splines, types=['H','He'])

    splines = [tmp_splines[i] if (i<7) else ZeroSpline() for i in range(12)]
    rhophis[n] = MEAM(splines = splines, types=['H','He'])

allPotentials = {'meams':meams, 'nophis':nophis, 'phionlys':phionlys,\
        'rhos':rhos, 'norhos':norhos, 'norhophis':norhophis, 'rhophis':rhophis}

# TODO: may be worth making a function to get LAMMPS eval for a set of
# atoms/potentials

print("Created %d potentials (%d main, %d subtypes)" % (7*N, N, 6*N))
