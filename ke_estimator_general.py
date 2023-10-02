import os
import ase
import numpy as np
import argparse
from ase.io import read, write

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help="task id")
parser.add_argument("--nbead", type=int, help="number of beads")
parser.add_argument("--temp", type=float, help="temperature")

args = parser.parse_args()
id = args.id
nbead = args.nbead
temp = args.temp
directory = "../task"+str(id)+"/rerun/"

lam = np.zeros(nbead)
for ibead in range(nbead):
  lam[ibead] = 4 * (np.sin(ibead * np.pi / nbead))**2

kB = 8.617343e-5
hplanck = 4.135667403e-3
mvv2e = 1.0364269e-4
hbar = hplanck / 2 / np.pi
beta = 1.0 / (kB * temp)
beta_np = 1.0 / (kB * temp) / nbead
omega_np = nbead / (hbar * beta) * mvv2e**0.5

print("Warning: this script assumes that the trajectory of each bead has the same number of frames.")
if not os.path.exists(directory+"mass_type_map.txt"):
    raise ValueError("Error: the file mass_type_map.txt does not exist. Please write the mass of each atom type in this file.")
mass_type_map = np.loadtxt(directory+"mass_type_map.txt", dtype=float)

atoms = ase.io.read(directory+"01.force", index=":")
atoms0 = atoms[0]
nframe = len(atoms)
positions0 = atoms0.get_positions()
types0 = atoms0.get_atomic_numbers()
natom = types0.shape[0]
ntype = np.unique(types0).shape[0]
mass_list = np.zeros(len(types0))
for itype in range(ntype):
    mass_list[types0==(itype+1)] = mass_type_map[itype]
coords = np.zeros([nbead, nframe, natom, 3])
forces = np.zeros([nbead, nframe, natom, 3])

# Read in the structure
for ibead in range(nbead):
    traj = ase.io.read(directory+"{:02d}.force".format(ibead+1), index=":")
    iframe = 0
    for atoms in traj:
        coords[ibead][iframe] = atoms.get_positions()
        box = atoms.get_cell()
        dx = coords[ibead][iframe] - positions0
        for idim in range(3):
            coords[ibead][iframe][dx[:, idim] > 0.5*box[idim][idim], idim] -= box[idim][idim]
            coords[ibead][iframe][dx[:, idim] < -0.5*box[idim][idim], idim] += box[idim][idim]
            # coords[ibead][iframe][:, idim] -= np.rint(dx[:, idim] / box[idim][idim]) * box[idim][idim
        dx = coords[ibead][iframe] - positions0
        dx_max = np.max(np.abs(dx))
        if dx_max > 0.5 * box[0][0]:
            raise ValueError("Warning: the maximum displacement is larger than half the box length.")
        forces[ibead][iframe] = atoms.get_forces()
        iframe += 1

# Calculate K_CV
xc = coords.mean(axis=0)
xcfs = (coords - xc) * forces
cv = ((xcfs.sum(axis=3)).sum(axis=2)).sum(axis=0)
kcv = (1.5*natom*kB*temp - 0.5/nbead*cv)/nbead

# Calculate K_prim
nmodes = nbead // 2
qnm = np.zeros(coords.shape)
qnmdummy = np.fft.rfft(coords, n=nbead, axis=0, norm="ortho")
qnmdummy[1:nmodes] *= 2**0.5
qnm[0] = qnmdummy[0].real
qnm[nmodes] = qnmdummy[nmodes].real
qnm[1:nmodes] = qnmdummy[1:nmodes].real
qnm[nbead-1:nmodes:-1] = qnmdummy[1:nmodes].imag
qnm2 = ((qnm**2).sum(axis=3)).reshape(nbead, nframe*natom)
lamqnm2 = ((np.matmul(np.diag(lam), qnm2)).sum(axis=0)).reshape(nframe, natom)
se = 0.5*omega_np**2*(lamqnm2*mass_list).sum(axis=1)
kpr = (1.5*natom*nbead*kB*temp - se/nbead)/nbead

np.save(directory+"se.npy", se)
np.save(directory+"kcv.npy", kcv)
np.save(directory+"kpr.npy", kpr)