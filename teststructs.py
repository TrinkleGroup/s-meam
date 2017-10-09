"""Creates test structures"""
import numpy as np
import ase.build
import testglobals

import lammpsTools

from ase import Atoms

nstructs = 0

# Builds dimers/trimers of H/He in vacuum; atom type has no impact
a0 = testglobals.a0
vac = testglobals.vac

dimers = []
trimers = []

r0 = a0/4.

dimer_aa = Atoms([1,1], positions=[[0,0,0],[r0,0,0]])
dimer_aa.center(vacuum=a0*2)
dimer_bb = Atoms([2,2], positions=[[0,0,0],[r0,0,0]])
dimer_ab = Atoms([1,2], positions=[[0,0,0],[r0,0,0]])

dimer_aa.center(vacuum=vac)
dimer_bb.center(vacuum=vac)
dimer_ab.center(vacuum=vac)
dimer_aa.center(vacuum=vac)
dimer_bb.center(vacuum=vac)
dimer_ab.center(vacuum=vac)

dimers.append(dimer_aa)
dimers.append(dimer_bb)
dimers.append(dimer_ab)

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

trimers.append(trimer_aaa)
trimers.append(trimer_bbb)
trimers.append(trimer_abb)
trimers.append(trimer_bab)
trimers.append(trimer_baa)
trimers.append(trimer_aba)

nsmall = len(trimers) + len(dimers)

# Builds H/He bulk structures with/without vacuum
orthogonal = ['sc','fcc','bcc','hcp','diamond']

bulk_vac = []              # bulk materials in a vacuum
bulk_periodic = []         # bulk with no vacuum

for s in orthogonal:
    atoms = ase.build.bulk('H',crystalstructure=s,a=a0)
    atoms = atoms.repeat((4,4,4))
    atoms.rattle()
    atoms.center(vacuum=0)
    atoms.set_pbc(True)
    atoms.set_chemical_symbols\
            (np.random.randint(1,3,size=len(atoms)))

    bulk_periodic.append(atoms)

    atoms.center(vacuum=a0*2)
    #print(atoms.get_cell())
    bulk_vac.append(atoms)

rhombohedral = ['fcc111', 'bcc111', 'hcp0001', 'diamond111']

bulk_vac_rhombo = []       # non-orthogonal bulk with vacuum
bulk_periodic_rhombo = []  # non-orthogonal bulk with no vacuum

fcc111 = ase.build.fcc111('H', size=(4,4,4),a=a0)
fcc111.set_pbc(True)
fcc111.rattle()
fcc111.center(vacuum=0)
bulk_periodic_rhombo.append(fcc111)
fcc111.center(vacuum=a0*2)
bulk_vac_rhombo.append(fcc111)

bcc111 = ase.build.bcc111('H', size=(4,4,4),a=a0)
bcc111.set_pbc(True)
bcc111.rattle()
bcc111.center(vacuum=0)
bulk_periodic_rhombo.append(bcc111)
bcc111.center(vacuum=a0*2)
bulk_vac_rhombo.append(bcc111)

hcp0001 = ase.build.hcp0001('H', size=(4,4,4),a=a0)
hcp0001.set_pbc(True)
hcp0001.rattle()
hcp0001.center(vacuum=0)
bulk_periodic_rhombo.append(hcp0001)
hcp0001.center(vacuum=a0*2)
bulk_vac_rhombo.append(hcp0001)

diamond111 = ase.build.diamond111('H', size=(4,4,4),a=a0)
diamond111.set_pbc(True)
diamond111.rattle()
diamond111.center(vacuum=0)
bulk_periodic_rhombo.append(diamond111)
diamond111.center(vacuum=a0*2)
bulk_vac_rhombo.append(diamond111)

nbig = len(bulk_periodic) + len(bulk_vac) + len(bulk_vac_rhombo) + len(bulk_periodic_rhombo)

print("Created %d total structures (%d dimers/trimers, %d bulk)" %(nbig+nsmall,\
    nsmall, nbig))
