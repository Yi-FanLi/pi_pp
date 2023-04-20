import numpy as np
from mpi4py import MPI
from deepmd.infer import DeepPot
import argparse

dp = DeepPot("/scratch/gpfs/yifanl/Work/pimd/h2o/scan_prl_lfz/frozen_model_compressed.pb")
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

ndiscard = 0000
#nsamp = 10001
nsamp = args.nsamp
nbatch=200
nloop=int(nsamp/nbatch)
zmmp_sc1 = np.zeros(nloop*nbatch)
zmmp_sc = np.zeros(nloop*nbatch)
zmmpO_sc1 = np.zeros(nloop*nbatch)
zmmpO_sc = np.zeros(nloop*nbatch)
se = np.zeros(nloop*nbatch)
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
mratio = np.zeros(natom)
mratio[idx_O] = (mO18/mO16)**0.5
mratio[idx_H] = (mD/mH)**0.5
mratio_bc = np.diag(mratio)+np.ones([natom, natom])-np.eye(natom)
mratio_xc_bc = np.diag(1-mratio)+np.ones([natom, natom])-np.eye(natom)

atype = np.zeros(natom)
atype[idx_H] = 1

cells = np.load(directory+"cells"+str(rank)+".npy")[ndiscard:ndiscard+nsamp]
for iloop in range(nloop):
  samp_idx=np.arange(iloop*nbatch, (iloop+1)*nbatch)
  coords_unmap_batch = coords_unmap[samp_idx]
  xus = np.array(comm.gather(coords_unmap_batch, root=0))
  if rank == 0:
    xc = xus.mean(axis=0)
  else:
    xc = None
  xc = comm.bcast(xc, root=0)
  coords_sc_batch = coords_unmap_batch[:, :, None, :, :] * mratio[None, None, :, :, None] + xc[:, None, :, :] * mratio_xc_bc[None, :, :, None]
  coords_sc_batch_feed = coords_sc_batch.reshape(nbatch*natom, natom*3)
  print("rank %d: coords_sc_batch_feed.shape"%(rank), coords_sc_batch_feed.shape)