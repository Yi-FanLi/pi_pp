import numpy as np
from mpi4py import MPI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help="task id")
args = parser.parse_args()

directory = "../task"+str(args.id)+"/"

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

ndiscard = 2000
nsamp = 3000
coords = np.load(directory+"coords"+str(rank)+".npy")[ndiscard:ndiscard+nsamp]
natom = coords.shape[1]
forces = np.load(directory+"forces"+str(rank)+".npy")[ndiscard:ndiscard+nsamp]
coords_unmap = np.load(directory+"coords_unmap"+str(rank)+".npy")[ndiscard:ndiscard+nsamp]

types=np.load(directory+'types'+str(rank)+'.npy')
idx = np.arange(natom)
idx_O = np.where(types[0]==1)[0]
idx_H = np.where(types[0]==2)[0]

nO = int(natom/3)
nH = 2*nO
mass = np.zeros(natom)
mass[idx_O] = 15.9994
mass[idx_H] = 1.00794

qnm = np.zeros([nbeads, nsamp, natom, 3])
xus = np.array(comm.gather(coords_unmap, root=0))
if rank == 0:
  xc = xus.mean(axis=0)
else:
  xc = None
xc = comm.bcast(xc, root=0)
xcf = (coords_unmap - xc) * forces
xcfs = np.array(comm.gather(xcf, root=0))

if rank == 0:
#  print(xus.shape)
  qnmdummy = np.fft.rfft(xus, n=nbeads, axis=0, norm="ortho")
  #print(qnmdummy[:, 0, 0, 0])
  qnmdummy[1:nmodes] *= 2**0.5
  qnm[0] = qnmdummy[0].real
  qnm[nmodes] = qnmdummy[nmodes].real
  qnm[1:nmodes] = qnmdummy[1:nmodes].real
  qnm[nbeads-1:nmodes:-1] = qnmdummy[1:nmodes].imag
  #print(np.diag(lam).shape)
  #print((qnm**2).sum(axis=3).shape)
  #print(qnm)
  qnm2 = ((qnm**2).sum(axis=3)).reshape(nbeads, nsamp*natom)
  #print(qnm2.shape)
  lamqnm2 = ((np.matmul(np.diag(lam), qnm2)).sum(axis=0)).reshape(nsamp, natom)
  se = 0.5*omega_np**2*(lamqnm2*mass).sum(axis=1)
#   print("se.shape:", se.shape)
#   print(xcfs.shape)
  cv = ((xcfs.sum(axis=3)).sum(axis=2)).sum(axis=0)
  kcv = (1.5*natom*kB*temp - 0.5/nbeads*cv)/nbeads
  kpr = (1.5*natom*nbeads*kB*temp - se/nbeads)/nbeads
#   print(kcv.shape)

  np.save(directory+"se.npy", se)
  np.save(directory+"kcv.npy", kcv)
  np.save(directory+"kpr.npy", kpr)
