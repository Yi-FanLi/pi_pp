import numpy as np
from mpi4py import MPI
import argparse
import copy
from time import time

line1 = "ITEM: TIMESTEP"
line3 = "ITEM: NUMBER OF ATOMS"
line5 = "ITEM: BOX BOUNDS pp pp pp"
line9 = "ITEM: ATOMS id type x y z"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbeads = comm.size
nmodes = nbeads // 2

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help="task id")
parser.add_argument("--nsamp", type=int, help="the number of samples")
parser.add_argument("--ndump", type=int, help="the number of samples")
args = parser.parse_args()

temp = 300.0
kB = 8.617343e-5
hplanck = 4.135667403e-3
mvv2e = 1.0364269e-4
hbar = hplanck / 2 / np.pi
beta = 1.0 / (kB * temp)
beta_np = 1.0 / (kB * temp) / nbeads
omega_np = nbeads / (hbar * beta) * mvv2e**0.5

ndiscard = 0000
#nsamp = 10001
nsamp = args.nsamp
ndump = args.ndump
nbatch=20
nloop=int(nsamp/nbatch)

directory = "../fractionation/task"+str(args.id)+"/"
outOxyz = directory+"%02d"%(rank+1)+"_Osc.xyz"
outHxyz = directory+"%02d"%(rank+1)+"_Hsc.xyz"
foutO = open(outOxyz, "a")
foutH = open(outHxyz, "a")
coords = np.load(directory+"coords"+str(rank)+".npy")[ndiscard:ndiscard+nsamp]
coords_unmap = np.load(directory+"coords_unmap"+str(rank)+".npy")[ndiscard:ndiscard+nsamp]
natom = coords.shape[1]

types=np.load(directory+'types'+str(rank)+'.npy')[ndiscard:ndiscard+nsamp]
idx = np.arange(natom)

idx_O = np.where(types[0]==1)[0]
idx_H = np.where(types[0]==2)[0]

nO = int(natom/3)
nH = 2*nO
mass = np.zeros(natom)
mass[idx_O] = 15.9994
mass[idx_H] = 1.00794
mH = 1.00794
mD = 2.01410
mO16 = 15.99491
mO18 = 17.99916
mratio = np.zeros(natom)
mratio[idx_O] = (mO18/mO16)**0.5
mratio[idx_H] = (mD/mH)**0.5

atype = np.zeros(natom)
atype[idx_H] = 1

cells = np.load(directory+"cells"+str(rank)+".npy")[ndiscard:ndiscard+nsamp]

t1 = time()
for iloop in range(nloop):
  samp_idx=np.arange(iloop*nbatch, (iloop+1)*nbatch)
  coords_unmap_batch = coords_unmap[samp_idx]
  cells_batch = cells[samp_idx]
  xus = np.array(comm.gather(coords_unmap_batch, root=0))
  if rank == 0:
    xc = xus.mean(axis=0)
  else:
    xc = None
  xc = comm.bcast(xc, root=0)
  coords_sc_O_batch = copy.deepcopy(coords_unmap_batch)
  coords_sc_O_batch[:, idx_O[0]] *= mratio[idx_O[0]]
  coords_sc_O_batch[:, idx_O[0]] += ((1-mratio[idx_O[0]]) * xc[:, idx_O[0]])

  coords_sc_H_batch = copy.deepcopy(coords_unmap_batch)
  coords_sc_H_batch[:, idx_H[0]] *= mratio[idx_H[0]]
  coords_sc_H_batch[:, idx_H[0]] += ((1-mratio[idx_H[0]]) * xc[:, idx_H[0]])

  for ibatch in range(nbatch):
    foutO.write(line1+"\n")
    foutH.write(line1+"\n")
    foutO.write("%d\n"%(samp_idx[ibatch]*ndump))
    foutH.write("%d\n"%(samp_idx[ibatch]*ndump))
    foutO.write(line3+"\n")
    foutH.write(line3+"\n")
    foutO.write("%d\n"%(natom))
    foutH.write("%d\n"%(natom))
    foutO.write(line5+"\n")
    foutH.write(line5+"\n")
    foutO.write("0 %.8f\n"%(cells_batch[ibatch][0]))
    foutH.write("0 %.8f\n"%(cells_batch[ibatch][0]))
    foutO.write("0 %.8f\n"%(cells_batch[ibatch][4]))
    foutH.write("0 %.8f\n"%(cells_batch[ibatch][4]))
    foutO.write("0 %.8f\n"%(cells_batch[ibatch][8]))
    foutH.write("0 %.8f\n"%(cells_batch[ibatch][8]))
    foutO.write(line9+"\n")
    foutH.write(line9+"\n")
    for iatom in range(natom):
      foutO.write("%d %d %.6f %.6f %.6f\n"%(iatom+1, types[samp_idx[ibatch]][iatom], coords_sc_O_batch[ibatch][iatom][0], coords_sc_O_batch[ibatch][iatom][1], coords_sc_O_batch[ibatch][iatom][2]))
      foutH.write("%d %d %.6f %.6f %.6f\n"%(iatom+1, types[samp_idx[ibatch]][iatom], coords_sc_H_batch[ibatch][iatom][0], coords_sc_H_batch[ibatch][iatom][1], coords_sc_H_batch[ibatch][iatom][2]))
  
  t2 = time()
  if rank == 0:
    print("Loop %d of %d, time: %.4f s.\n"%(iloop, nloop, t2-t1))