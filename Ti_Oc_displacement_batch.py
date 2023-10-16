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
parser.add_argument("--rcut", type=float, default=2.5, help="cutoff radius")
parser.add_argument("--nframe", type=int, default=None, help="number of frames to be read")
parser.add_argument("--nbatch", type=int, default=100, help="number of frames in a batch")
parser.add_argument("--nevery", type=int, default=1, help="every this step to read a frame")
parser.add_argument("--ndiscard", type=int, default=0, help="number of frames to discard")

args = parser.parse_args()
id = args.id
# nbead = args.nbead
temp = args.temp
rcut = args.rcut
nframe = args.nframe
nbatch = args.nbatch
nevery = args.nevery

ndiscard = args.ndiscard
directory = "../task"+str(id)+"/"
mass_type_map = np.loadtxt(directory+"mass_type_map.txt", dtype=float)

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

atoms0 = atoms[0]
if nframe is None:
    nframe = len(atoms)
nloop = int(nframe/nbatch)
nframe_write = nloop * nbatch
positions0 = atoms0.get_positions()
types0 = atoms0.get_atomic_numbers()

idx_Ti = np.where(types0==2)[0]
idx_O = np.where(types0==3)[0]

# print("idx_Ti: ", idx_Ti)
# print("idx_O: ", idx_O)

type_numbers = [type_to_atomic_number[t] for t in types0]
atoms0.set_atomic_numbers(type_numbers)
natom = types0.shape[0]
ntype = np.unique(types0).shape[0]
mass_list = np.zeros(len(types0))
for itype in range(ntype):
    mass_list[types0==(itype+1)] = mass_type_map[itype]

d_Ti_Oc = np.empty([nbead, nframe_write, idx_Ti.shape[0], 3])
d_Ti_Oc_bead = np.empty([nframe_write, idx_Ti.shape[0], 3])
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
    positionsO = positions[:, idx_O]
    positionsTi = positions[:, idx_Ti]
    dist_batch = positionsTi[:, :, None] - positionsO[:, None, :]
    dist_pbc = (dist_batch / prds[:, None, None] - np.floor(dist_batch / prds[:, None, None] + 0.5)) * prds[:, None, None]
    dist_r = np.sqrt((dist_pbc**2).sum(axis=3))
    TiO6_idx = np.where(dist_r <= rcut)
    d_TiO6 = dist_pbc[TiO6_idx].reshape([nbatch, idx_Ti.shape[0], 6, 3])
    d_Ti_Oc_bead[output_idx] = d_TiO6.mean(axis=2)
    tloopend = time()
    print("Bead {b} loop {l} costs {t} s.".format(b=ibead, l=iloop, t=tloopend-tloopstart))

tend = time()
print("Bead {b} costs {t} s.".format(b=ibead, t=tend-tstart))

comm.Gather(sendbuf=d_Ti_Oc_bead, recvbuf=d_Ti_Oc, root=0)
if rank==0:
    np.save(directory+"d_Ti_Oc.npy", d_Ti_Oc)
