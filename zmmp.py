import numpy as np
from mpi4py import MPI
#from deepmd.infer import DeepPot
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbeads = comm.size
nmodes = nbeads // 2

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help="task id")
parser.add_argument("--nsamp", type=int, help="the number of samples")
args = parser.parse_args()

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

ndiscard = 0000
#nsamp = 10001
nsamp = args.nsamp
directory = "../fractionation/task"+str(args.id)+"/"
coords = np.load(directory+"coords"+str(rank)+".npy")[ndiscard:ndiscard+nsamp]
natom = coords.shape[1]

types=np.load(directory+'types'+str(rank)+'.npy')[ndiscard:ndiscard+nsamp]
idx = np.arange(natom)
#idx_O = np.arange(0, natom, 3)
#idx_H = np.delete(idx, idx_O)
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
#if rank==0:
#  print(mass)

#dp = DeepPot('model-compress.pb')
atype = np.zeros(natom)
atype[nO:natom] = 1

#coords = np.zeros([nsamp, natom, 3])
#cells = np.zeros([nsamp, 9])
#coords_unmap = np.zeros([nsamp, natom, 3])
coords_unmap = np.load(directory+"coords_unmap"+str(rank)+".npy")[ndiscard:ndiscard+nsamp]
cells = np.load(directory+"cells"+str(rank)+".npy")[ndiscard:ndiscard+nsamp]
qnm = np.zeros([nbeads, nsamp, natom, 3])
images = np.zeros([nsamp, natom, 3])
#Vp = np.zeros(nsamp)

print("rank: %d"%(rank), coords_unmap.shape)
xus = np.array(comm.gather(coords_unmap, root=0))
if rank == 0:
  xc = xus.mean(axis=0)
else:
  xc = None
xc = comm.bcast(xc, root=0)
#print(coords_unmap.shape)
#print(xc.shape)

#xp = coords
#xp[:, 1, :] = xc[:, 1, :] + (mD/mH)**0.5*(coords_unmap[:, 1, :] - xc[:, 1, :])
#for i in range(nsamp):
#  e, f, v = dp.eval(xp[i], cells[i], atype)
#  Vp[i] = e
#Vp_beads = np.array(comm.gather(Vp, root=0))

#if rank == 0:
#  print(Vp_beads)

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
  #print(wqnm2.shape)
  zmmp1 = np.exp(-0.5*beta_np*omega_np**2*(mD-mH)*lamqnm2[:, idx_H[0]])
  zmmp = (np.exp(-0.5*beta_np*omega_np**2*(mD-mH)*lamqnm2[:, idx_H])).mean(axis=1)

  zmmpO1 = np.exp(-0.5*beta_np*omega_np**2*(mO18-mO16)*lamqnm2[:, idx_O[0]])
  zmmpO = (np.exp(-0.5*beta_np*omega_np**2*(mO18-mO16)*lamqnm2[:, idx_O])).mean(axis=1)

  se = 0.5*omega_np**2*(lamqnm2*mass).sum(axis=1)
  np.save(directory+"zmmp1.npy", zmmp1)
  np.save(directory+"zmmp.npy", zmmp)

  np.save(directory+"zmmpO1.npy", zmmpO1)
  np.save(directory+"zmmpO.npy", zmmpO)

  np.save(directory+"se.npy", se)
  #print(zmmp)
  #print(se)
  #se = 0.5*(mass*((np.matmul(np.diag(lam), (qnm**2).sum(axis=3)).sum(axis=0)))).sum(axis=2)
#if rank == 0:
  #print(qnm)
