# import sys
# sys.path.append('../src')

import numpy as np
import ase.build
from .globalVars import a0, r0, vac

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ase import Atoms

################################################################################

"""Structures are stored in dictionaries with key:val = name:Atoms. All 
structures are centered in their cell. Bulk structures are 'rattled' off of 
their default positions. Structure in vacuum have equal amounts of vacuum in 
all directions."""

dimers                  = {}
trimers                 = {}
bulk_vac_ortho          = {}
bulk_periodic_ortho     = {}
bulk_vac_rhombo         = {}
bulk_periodic_rhombo    = {}

################################################################################

"""Dimers"""

dimer_aa = Atoms([1,1], positions=[[0,0,0],[r0,0,0]])
dimer_bb = Atoms([2,2], positions=[[0,0,0],[r0,0,0]])
dimer_ab = Atoms([1,2], positions=[[0,0,0],[r0,0,0]])

dimer_aa.center(vacuum=vac)
dimer_bb.center(vacuum=vac)
dimer_ab.center(vacuum=vac)

dimer_aa.set_pbc(True)
dimer_bb.set_pbc(True)
dimer_ab.set_pbc(True)

dimers['aa'] = dimer_aa
dimers['bb'] = dimer_bb
dimers['ab'] = dimer_ab

################################################################################

"""Trimers"""

trimer_aaa = Atoms([1,1,1],
        positions=[[0,0,0],[r0,0,0],[r0/2,np.sqrt(3)*r0/2,0]])
trimer_bbb = Atoms([2,2,2],
        positions=[[0,0,0],[r0,0,0],[r0/2,np.sqrt(3)*r0/2,0]])
trimer_abb = Atoms([1,2,2],
        positions=[[0,0,0],[r0,0,0],[r0/2,np.sqrt(3)*r0/2,0]])
trimer_bab = Atoms([2,1,2],
        positions=[[0,0,0],[r0,0,0],[r0/2,np.sqrt(3)*r0/2,0]])
trimer_baa = Atoms([2,1,1],
        positions=[[0,0,0],[r0,0,0],[r0/2,np.sqrt(3)*r0/2,0]])
trimer_aba = Atoms([1,2,1],
        positions=[[0,0,0],[r0,0,0],[r0/2,np.sqrt(3)*r0/2,0]])

trimer_aaa.set_pbc(True)
trimer_bbb.set_pbc(True)
trimer_abb.set_pbc(True)
trimer_bab.set_pbc(True)
trimer_baa.set_pbc(True)
trimer_aba.set_pbc(True)

trimer_aaa.center(vacuum=vac)
trimer_bbb.center(vacuum=vac)
trimer_abb.center(vacuum=vac)
trimer_bab.center(vacuum=vac)
trimer_baa.center(vacuum=vac)
trimer_aba.center(vacuum=vac)

trimers['aaa'] = trimer_aaa
trimers['bbb'] = trimer_bbb
trimers['abb'] = trimer_abb
trimers['bab'] = trimer_bab
trimers['baa'] = trimer_baa
trimers['aba'] = trimer_aba

################################################################################

"""Orthogonal bulk structures"""

type1 = ase.build.bulk('H', crystalstructure='fcc', a=a0, orthorhombic=True)
type1 = type1.repeat((4,4,4))
type1.rattle()
type1.center(vacuum=0)
type1.set_pbc(True)
type1.set_chemical_symbols(np.ones(len(type1), dtype=int))

type2 = ase.build.bulk('H', crystalstructure='fcc', a=a0, orthorhombic=True)
type2 = type2.repeat((4,4,4))
type2.rattle()
type2.center(vacuum=0)
type2.set_pbc(True)
type2.set_chemical_symbols(np.ones(len(type1), dtype=int)*2)

mixed = ase.build.bulk('H', crystalstructure='fcc', a=a0, orthorhombic=True)
mixed = mixed.repeat((4,4,4))
mixed.rattle()
mixed.center(vacuum=0)
mixed.set_pbc(True)
mixed.set_chemical_symbols(np.random.randint(1,3,size=len(mixed)))

bulk_periodic_ortho['bulk_periodic_ortho_type1'] = type1
bulk_periodic_ortho['bulk_periodic_ortho_type2'] = type2
bulk_periodic_ortho['bulk_periodic_ortho_mixed'] = mixed

# TODO: check that assignment creates a copy; are bulk periodic still periodic?

type1.center(vacuum=vac)
type2.center(vacuum=vac)
mixed.center(vacuum=vac)

bulk_vac_ortho['bulk_vac_ortho_type1'] = type1
bulk_vac_ortho['bulk_vac_ortho_type2'] = type2
bulk_vac_ortho['bulk_vac_ortho_mixed'] = mixed

################################################################################

"""Rhombohedral bulk structures"""

type1 = ase.build.fcc111('H', size=(4,4,4),a=a0)
type1.rattle()
type1.center(vacuum=0)
type1.set_pbc(True)
type1.set_chemical_symbols(np.ones(len(type1), dtype=int))

# TODO: does set_chemical_symbols() change chemical formula?

type2 = ase.build.fcc111('H', size=(4,4,4),a=a0)
type2.rattle()
type2.center(vacuum=0)
type2.set_pbc(True)
type2.set_chemical_symbols(np.ones(len(type2), dtype=int)*2)

mixed = ase.build.fcc111('H', size=(4,4,4),a=a0)
mixed.rattle()
mixed.center(vacuum=0)
mixed.set_pbc(True)
mixed.set_chemical_symbols(np.random.randint(1,3,size=len(mixed)))

bulk_periodic_rhombo['bulk_periodic_rhombo_type1'] = type1
bulk_periodic_rhombo['bulk_periodic_rhombo_type2'] = type2
bulk_periodic_rhombo['bulk_periodic_rhombo_mixed'] = mixed

type1.center(vacuum=vac)
type2.center(vacuum=vac)
mixed.center(vacuum=vac)

bulk_vac_rhombo['bulk_vac_rhombo_type1'] = type1
bulk_vac_rhombo['bulk_vac_rhombo_type2'] = type2
bulk_vac_rhombo['bulk_vac_rhombo_mixed'] = mixed

################################################################################
allstructs = {**dimers, **trimers, **bulk_vac_ortho, **bulk_periodic_ortho,\
                 **bulk_vac_rhombo, **bulk_periodic_rhombo}
# key = 'bulk_vac_ortho_type1'
# allstructs = {key:bulk_vac_ortho[key]}
# key = 'ab'
# allstructs = {key:dimers[key]}
#allstructs = {'twotrips':lammpsTools.atoms_from_file('data.twotrips',types=[
#    'H','He'])}

#lammpsTools.atoms_to_LAMMPS_file('data.dimerBB', dimers[key])

nsmall = len(dimers) + len(trimers)
nbig = len(bulk_periodic_ortho) + len(bulk_vac_ortho) + len(bulk_vac_rhombo) +\
        len(bulk_periodic_rhombo)

#print("Created %d total structures (%d dimers/trimers, %d bulk)" %(
## nbig+nsmall,\#
            #nsmall, nbig))