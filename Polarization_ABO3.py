import numpy as np
import ase
import numpy as np
import argparse
from ase.io import read, write
import ase.geometry
from time import time
from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help="task id")
# parser.add_argument("--nbead", type=int, default=1, help="number of beads")
parser.add_argument("--temp", type=float, help="temperature")
parser.add_argument("--rcutTiO6", type=float, default=2.5, help="cutoff radius for TiO6")
parser.add_argument("--rcutTiSr8", type=float, default=3.8, help="cutoff radius for TiSr8")
parser.add_argument("--nframe", type=int, default=None, help="number of frames to be read")
parser.add_argument("--nbatch", type=int, default=100, help="number of frames in a batch")
parser.add_argument("--nevery", type=int, default=1, help="every this step to read a frame")
parser.add_argument("--ndiscard", type=int, default=0, help="number of frames to discard")

args = parser.parse_args()
id = args.id
# nbead = args.nbead
temp = args.temp
rcutTiO6 = args.rcutTiO6
rcutTiSr8 = args.rcutTiSr8
nframe = args.nframe
nbatch = args.nbatch
nevery = args.nevery

ndiscard = args.ndiscard
directory = "../task"+str(id)+"/"

elementary_charge = 1.60217662e-19 # C
Angstrom = 1e-10 # m
unit_converter = elementary_charge / Angstrom**2 # C/m^2

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
nbead = size
ibead = rank

atoms = ase.io.read(directory+"{:02d}.xyz".format(ibead+1), index=":")

type_to_atomic_number = {
    1: 38, # Sr has atomic number 38
    2: 22,  # Ti has atomic number 22
    3: 8   # O has atomic number 8
}

ZO1 = 5.66 # O1 are the O atoms in the TiO2 along the z direction
ZO2 = 2.00 # O2 are the O atoms in the TiO4 in the x-y plane
ZSr = 2.54 # Sr are the Sr atoms in the TiSr8
ZTi = 7.12 # Ti are the Ti atoms

atoms0 = atoms[0]
if nframe is None:
    nframe = len(atoms)
nloop = int(nframe/nbatch)
nframe_write = nloop * nbatch
positions0 = atoms0.get_positions()
types0 = atoms0.get_atomic_numbers()

idx_Sr = np.where(types0==1)[0]
idx_Ti = np.where(types0==2)[0]
idx_O = np.where(types0==3)[0]

# print("idx_Ti: ", idx_Ti)
# print("idx_O: ", idx_O)

type_numbers = [type_to_atomic_number[t] for t in types0]
atoms0.set_atomic_numbers(type_numbers)
natom = types0.shape[0]
ntype = np.unique(types0).shape[0]

dipole_z = np.empty([nbead, nframe_write, idx_Ti.shape[0]])
dipole_z_bead = np.empty([nframe_write, idx_Ti.shape[0]])
# coords = np.zeros([nbead, nframe, natom, 3])
# print(atoms0.get_chemical_symbols())

# tstart = time()
# for ibead in range(nbead):
tstart = time()
for iloop in range(nloop):
    tloopstart = time()
    batch_idx = np.arange(iloop*nbatch*nevery, (iloop+1)*nbatch*nevery, nevery)+ndiscard
    output_idx = np.arange(iloop*nbatch, (iloop+1)*nbatch)
    # print(batch_idx)
    # traj = ase.io.read(directory+"{:02d}.xyz".format(ibead+1), index=batch_idx.tolist())
    traj = ase.io.read(directory+"{:02d}.xyz".format(ibead+1), index="%d:%d:%d"%(iloop*nbatch*nevery+ndiscard, (iloop+1)*nbatch*nevery+ndiscard, nevery))
    positions = np.zeros([nbatch, natom, 3])
    prds = np.zeros([nbatch, 3])
    iframe = 0
    for atoms in traj:
        positions[iframe] = atoms.get_positions()
        for dd in range(3):
            prds[iframe][dd] = atoms.get_cell()[dd][dd]
        iframe += 1

    positionsSr = positions[:, idx_Sr]
    positionsTi = positions[:, idx_Ti]
    positionsO = positions[:, idx_O]

    # picking out TiO4 in the x-y plane and TiO2 in the z direction
    dist_TiO_batch = positionsTi[:, :, None] - positionsO[:, None, :]
    dist_TiO_pbc = (dist_TiO_batch / prds[:, None, None] - np.floor(dist_TiO_batch / prds[:, None, None] + 0.5)) * prds[:, None, None]
    dist_TiO_r = np.sqrt((dist_TiO_pbc**2).sum(axis=3))

    TiO4_xy_idx = np.where((dist_TiO_r <= rcutTiO6) & (np.abs(dist_TiO_pbc[:, :, :, 2]) < 1.0))
    TiO2_z_idx = np.where((dist_TiO_r <= rcutTiO6) & (np.abs(dist_TiO_pbc[:, :, :, 2]) > 1.5))


    d_TiO4_xy = dist_TiO_pbc[TiO4_xy_idx].reshape([nbatch, idx_Ti.shape[0], 4, 3])
    d_TiO2_z = dist_TiO_pbc[TiO2_z_idx].reshape([nbatch, idx_Ti.shape[0], 2, 3])

    # picking out TiSr8
    dist_TiSr_batch = positionsTi[:, :, None] - positionsSr[:, None, :]
    dist_TiSr_pbc = (dist_TiSr_batch / prds[:, None, None] - np.floor(dist_TiSr_batch / prds[:, None, None] + 0.5)) * prds[:, None, None]
    dist_TiSr_r = np.sqrt((dist_TiSr_pbc**2).sum(axis=3))

    TiSr8_idx = np.where(dist_TiSr_r <= rcutTiSr8)

    d_TiSr8 = dist_TiSr_pbc[TiSr8_idx].reshape([nbatch, idx_Ti.shape[0], 8, 3])


    # print(d_TiO4_xy.shape)
    # print(d_TiO2_z.shape)
    # print(d_TiSr8.shape)

    # calculate the dipole moment

    dipole_z_bead[output_idx] = (ZO1 * d_TiO2_z[:, :, :, 2].mean(axis=2) + ZO2 * 2 * d_TiO4_xy[:, :, :, 2].mean(axis=2) - ZSr * d_TiSr8[:, :, :, 2].mean(axis=2)) / (prds[:, 0] * prds[:, 1] * prds[:, 2])[:, None] * idx_Ti.shape[0] * unit_converter

    # d_Ti_Oc_bead[output_idx] = d_TiO6.mean(axis=2)
    tloopend = time()
    print("Bead {b} loop {l} costs {t} s.".format(b=ibead, l=iloop, t=tloopend-tloopstart))

tend = time()
print("Bead {b} costs {t} s.".format(b=ibead, t=tend-tstart))

comm.Gather(sendbuf=dipole_z_bead, recvbuf=dipole_z, root=0)
if rank==0:
    np.save(directory+"dipole_z.npy", dipole_z)