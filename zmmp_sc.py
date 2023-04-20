import numpy as np
from mpi4py import MPI
from deepmd.infer import DeepPot
import argparse
import copy
from time import time

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
nbatch=10
nloop=int(nsamp/nbatch)
zmmp_sc1 = np.zeros(nloop*nbatch)
zmmp_sc = np.zeros(nloop*nbatch)
zmmpO_sc1 = np.zeros(nloop*nbatch)
zmmpO_sc = np.zeros(nloop*nbatch)
se = np.zeros(nloop*nbatch)
directory = "../h2o/task"+str(args.id)+"/"
coords = np.load(directory+"coords"+str(rank)+".npy")[ndiscard:ndiscard+nsamp]
coords_unmap = np.load(directory+"coords_unmap"+str(rank)+".npy")[ndiscard:ndiscard+nsamp]
natom = coords.shape[1]
# print("natom = %d"%(natom))

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
# if rank == 0:
e_unscaled = np.load(directory+"pote"+str(rank)+".npy")[ndiscard:ndiscard+nsamp]
ros(nloop*nbatch)
e_sc_O = np.zeros(nloop*nbatch)

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
  # coords_sc_batch = coords_unmap_batch[:, None, :, :] * mratio_bc[None, :, :, None] + xc[:, None, :, :] * mratio_xc_bc[None, :, :, None]
  # coords_sc_batch_feed = coords_sc_batch.reshape(nbatch*natom, natom*3)
  coords_sc_O_batch = copy.deepcopy(coords_unmap_batch)
  # print(coords_sc_O_batch[:, idx_O[0]])
  coords_sc_O_batch[:, idx_O[0]] *= mratio[idx_O[0]]
  # print("after xmratio, ")
  # print(coords_sc_O_batch[:, idx_O[0]])
  coords_sc_O_batch[:, idx_O[0]] += ((1-mratio[idx_O[0]]) * xc[:, idx_O[0]])
  # print("finally, ")
  # print(coords_sc_O_batch[:, idx_O[0]])

  coords_sc_H_batch = copy.deepcopy(coords_unmap_batch)
  coords_sc_H_batch[:, idx_H[0]] *= mratio[idx_H[0]]
  coords_sc_H_batch[:, idx_H[0]] += ((1-mratio[idx_H[0]]) * xc[:, idx_H[0]])

  # print("xO:")
  # print(coords_unmap_batch[:, idx_O[0]])
  # print("xO, scaled:")
  # print(coords_sc_O_batch[:, idx_O[0]])

  # print("xH:")
  # print(coords_unmap_batch[:, idx_H[0]])
  # print("xH, scaled:")
  # print(coords_sc_H_batch[:, idx_H[0]])

  e, f, v = dp.eval(coords_unmap_batch.reshape(nbatch, natom*3), cells_batch, atype)
  e_unscaled[samp_idx] = e.flatten()
  e_sc_O_batch, f_O_sc, v_O_sc = dp.eval(coords_sc_O_batch.reshape(nbatch, natom*3), cells_batch, atype)
  e_sc_O[samp_idx] = e_sc_O_batch.flatten()
  e_sc_H_batch, f_H_sc, v_H_sc = dp.eval(coords_sc_H_batch.reshape(nbatch, natom*3), cells_batch, atype)
  e_sc_H[samp_idx] = e_sc_H_batch.flatten()
  
  t2 = time()
  if rank == 0:
    print("Loop %d of %d, time: %.4f s.\n"%(iloop, nloop, t2-t1))

e_unscaled_beads = np.array(comm.gather(e_unscaled, root=0))
e_sc_O_beads = np.array(comm.gather(e_sc_O, root=0))
e_sc_H_beads = np.array(comm.gather(e_sc_H, root=0))
if rank == 0:
  e_unscaled_sum = e_unscaled_beads.sum(axis=0)
  e_sc_O_sum = e_sc_O_beads.sum(axis=0)
  e_sc_H_sum = e_sc_H_beads.sum(axis=0)
  zmmp_sc_O = np.exp(beta_np*(e_unscaled_sum-e_sc_O_sum))
  zmmp_sc_H = np.exp(beta_np*(e_unscaled_sum-e_sc_H_sum))
  np.save(directory+"zmmp_sc_O.npy", zmmp_sc_O)
  np.save(directory+"zmmp_sc_H.npy", zmmp_sc_H)
  # np.save("e_sc_O_sum.npy", e_sc_O_sum)
  # np.save("e_sc_H_sum.npy", e_sc_H_sum)
  # cells_batch_feed = np.array([cells_batch for i in range(natom)]).reshape(nbatch*natom, 9)
  # e_sc = np.zeros(nbatch*natom)
  # for jloop in range(natom):
  #   minibatch_idx=np.arange(jloop*nbatch, (jloop+1)*nbatch)
  #   coords_sc_minibatch_feed = coords_sc_batch_feed[minibatch_idx]
  #   cells_minibatch_feed = cells_batch_feed[minibatch_idx]
  #   e_sc_minibatch, f_sc_minibatch, v_sc_minibatch = dp.eval(coords_sc_minibatch_feed, cells_minibatch_feed, atype)
  #   e_sc[minibatch_idx] = e_sc_minibatch.flatten()
  #   t3 = time()
  #   print("Atom %d of %d, loop %d, time: %.4f s.\n"%(jloop, natom, iloop, t3-t1))
  # # e_sc, f_sc, v_sc = dp.eval(coords_sc_batch_feed, cells_batch_feed, atype)
  # t2 = time()
  # print("Loop %d of %d, time: %.4f s.\n"%(iloop, nloop, t2-t1))
#   e_sc_beads = np.array(comm.gather(e_sc, root=0))
#   if rank == 0:
#     e_sc_sum = e_sc_beads.sum(axis=0)
#     e_sc_atoms[samp_idx] = e_sc_sum

# if rank == 0:
#   np.save("e_sc_atoms.npy", e_sc_atoms)