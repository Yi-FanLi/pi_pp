import numpy as np
from time import time
from mpi4py import MPI
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help="task id")
args = parser.parse_args()

directory = "../task"+str(args.id)+"/"
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbeads = comm.size

t0 = time()

coords=np.load(directory+'coords'+str(rank)+'.npy')
types=np.load(directory+'types'+str(rank)+'.npy')
cells=np.load(directory+'cells'+str(rank)+'.npy')

ndiscard=10001
#ndiscard=0
nsamp=10000
coords=coords[ndiscard:ndiscard+nsamp]
types=types[ndiscard:ndiscard+nsamp]
cells=cells[ndiscard:ndiscard+nsamp]
#nsamp=coords.shape[0]
natom=coords.shape[1]
nO=int(natom/3)
nH=2*int(natom/3)

t2 = time()
if rank==0:
  print("Reading samples costs %.4f s."%(t2-t0))

rcut=12
nbin=1000
dr=rcut/nbin
r_array=np.linspace(0, rcut, num=nbin, endpoint=False)+0.5*dr

#idx_O = np.where(types[0]==1)[0]
idx_H = np.where(types[0]==2)[0]
#coords_O = coords[:, idx_O]
coords_H = coords[:, idx_H]
#dist=coords_O[:, None, :]-coords_O[:, :, None]
prds=(cells[:, [0,4,8]]).reshape(nsamp, 1, 1, 3)

#print(prds[0])
#print(dist[0][0][1])
#print(dist_pbc[0][0][1])

dists_array = np.zeros([nsamp, int(nH*(nH-1)/2)])
g_r_array = np.zeros([nsamp, nbin])

nbatch = 100
nloop = int(nsamp/nbatch)
for iloop in range(nloop):
  t1 = time()
  #coords_O_batch = coords_O[iloop*nbatch:(iloop+1)*nbatch]
  coords_H_batch = coords_H[iloop*nbatch:(iloop+1)*nbatch]
  #dist_batch = coords_O_batch[:, None, :] - coords_O_batch[:, :, None]
  dist_batch = coords_H_batch[:, None, :] - coords_H_batch[:, :, None]
  #dist_batch = dist[iloop*nbatch:(iloop+1)*nbatch]
  prds_batch = prds[iloop*nbatch:(iloop+1)*nbatch]
  dist_pbc=(dist_batch/prds_batch-np.floor(dist_batch/prds_batch+0.5))*prds_batch
  dist_r=np.sqrt((dist_pbc**2).sum(axis=3))
  
  t3 = time()
  if rank==0:
    print("Loop %d: computing pbc distances costs %.4f s."%(iloop, t3-t1))
  for ibatch in range(nbatch):
    isamp = iloop*nbatch + ibatch
    dists_array[isamp] = dist_r[ibatch][np.triu_indices(nH, 1)]
  #print(dist_r[isamp][0][1])
  #print(dists_array[isamp])
    Vol = cells[isamp][0]*cells[isamp][4]*cells[isamp][8]
    hist_r = np.histogram(dists_array[isamp], bins=nbin, range=(0, rcut), density=False)
  #print(hist_r)
  #print(hist_r[0].sum())
  #print(r_array)
    g_r_array[isamp] = 2*hist_r[0]/4/np.pi/r_array**2/dr/nH/(nH-1)*Vol
    if rank==0:
      if (isamp+1)%100==0:
        t4 = time()
        print("Computing rdf for %d samples costs %.4f s."%(isamp+1, t4-t0))

g_r_bead = g_r_array.mean(axis=0)
#if rank==0:
#  g_r_beads=None
g_r_beads = comm.gather(g_r_bead, root=0)
#if rank==0:
#  print(g_r_bead)
#print(g_r.sum())

#hist_r=np.histogram(dists_array.reshape(-1, ), bins=nbin, range=(0, rcut), density=False)
#rho_r=hist_r[0]/nsamp
#r_array=hist_r[1][1:]+0.5*(hist_r[1][1]-hist_r[1][0])
#g_r=rho_r/4/np.pi/r_array**2/dr
if rank==0:
  g_r = (np.array(g_r_beads, dtype="float").reshape(nbeads, nbin)).mean(axis=0)
  #print(g_r.shape)
  np.save(directory+"ghh.npy", np.c_[r_array, g_r])
  #plt.xlim(2, 7.5)
  #plt.xticks(np.arange(2, 8, 1))
  #plt.plot(r_array, g_r)
  #plt.show()
#r_array=np.linspace(0, 9, 300, endpoint=True)
#print(rho_r)
#print(r_array)
#print(r_array)
