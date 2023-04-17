import os
import numpy as np
from time import time
from mpi4py import MPI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help="task id")
parser.add_argument("--natom", type=int, help="number of atoms")
parser.add_argument("--nsamp", type=int, help="the number of samples")
args = parser.parse_args()

directory = "../fractionation/task"+str(args.id)+"/"
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbeads = comm.size
nmodes = nbeads // 2

temp = 300.0
kB = 8.617343e-5
hplanck = 4.135667403e-3
mvv2e = 1.0364269e-4
hbar = hplanck / 2 / np.pi
beta = 1.0 / (kB * temp)
beta_np = 1.0 / (kB * temp) / nbeads
omega_np = nbeads / (hbar * beta) * mvv2e**0.5

lam = np.zeros(nbeads)
for ibead in range(nbeads):
  lam[ibead] = 4 * (np.sin(ibead * np.pi / nbeads))**2
#if rank==0:
#  print(lam)

#ndiscard = 000
nsamp = args.nsamp
natom = args.natom

nO = int(natom/3)
nH = nO*2
mass = np.zeros(natom)
mass[:nO] = 15.9994
mass[nO:nO+nH] = 1.00794
mH = 1.00794
mD = 2.01410
#if rank==0:
#  print(mass)

#atype = np.zeros(natom)
#atype[nO:natom] = 1
#atypes = np.array([atype for i in range(nsamp)])
#print(atypes.shape)

coords = np.zeros([nsamp, natom, 3])
forcenms = np.zeros([nsamp, natom, 3])
types = np.zeros([nsamp, natom])
cells = np.zeros([nsamp, 9])
coords_unmap = np.zeros([nsamp, natom, 3])
qnm = np.zeros([nbeads, nsamp, natom, 3])
images = np.zeros([nsamp, natom, 3])
vels = np.zeros([nsamp, natom, 3])
Vp = np.zeros(nsamp)

#log_file = np.loadtxt("log."+str(rank), skiprows=128, max_rows=nsamp)
#V = log_file[:, 3]

t1 = time()
with open(directory+"%02d"%(rank+1)+".xyz", "r") as f:
  #print(f.readline())
  #for i in range(ndiscard):
  #  for j in range(natom+9):
  #    f.readline()
  for i in range(nsamp):
    for j in range(5):
      f.readline() 
    xlohi = np.array(f.readline().split(), dtype="float")
    xprd = xlohi[1] - xlohi[0]
    ylohi = np.array(f.readline().split(), dtype="float")
    yprd = ylohi[1] - ylohi[0]
    zlohi = np.array(f.readline().split(), dtype="float")
    zprd = zlohi[1] - zlohi[0]
    cells[i] = np.diag(np.array([xprd, yprd, zprd])).reshape(1, -1)
    f.readline()
    for j in range(natom):
      line = f.readline().split()
      #print(line)
      types[i][j] = int(line[1])
      coords[i][j] = np.array(line[2:5], dtype="float")
      #vels[i][j] = np.array(line[5:8], dtype="float")
      images[i][j] = np.array(line[5:8], dtype="int")
      forcenms[i][j] = np.array(line[8:11], dtype="float")
    coords_unmap[i] = coords[i] + images[i]*np.array([xprd, yprd, zprd])
    if i%100 == 0:
      if rank == 0:
        t3 = time()
        print("Rank = %d: reading %d samples costs %.4f s.\n"%(rank, (i+1), t3-t1))
t2 = time()
print("Rank = %d: reading %d samples costs %.4f s.\n"%(rank, nsamp, t2-t1))
np.save(directory+"types"+str(rank)+".npy", types)
np.save(directory+"coords"+str(rank)+".npy", coords)
#np.save(directory+"forcenms"+str(rank)+".npy", forcenms)
#np.save(directory+"vels"+str(rank)+".npy", vels)
np.save(directory+"coords_unmap"+str(rank)+".npy", coords_unmap)
np.save(directory+"cells"+str(rank)+".npy", cells)
