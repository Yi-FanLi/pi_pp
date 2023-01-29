# inverse normal mode transformation on forces
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

forcenms = np.load(directory+"forcenms"+str(rank)+".npy")
print("forcenms loaded")
nsamp = forcenms.shape[0]
natom = forcenms.shape[1]
nbatch = 5000
forcenms_batch = forcenms[:nbatch]
nsamp = nbatch
fnms = np.array(comm.gather(forcenms_batch, root=0))

if rank==0:
  fnms_complex = np.zeros([nmodes + 1, nsamp, natom, 3], complex)
  fnms_complex[0] = fnms[0]
  fnms_complex[nmodes] = fnms[nmodes]
#   print(fnms.shape)
#   print(fnms_complex.shape)
  fnms_complex[1:nmodes].real = fnms[1:nmodes]
  fnms_complex[1:nmodes].imag = fnms[nbeads-1:nmodes:-1]
  fnms_complex[1:nmodes] /= 2**0.5
  forces = np.fft.irfft(fnms_complex, n=nbeads, axis=0, norm="ortho")
  for ib in range(nbeads):
    np.save(directory+"forces"+str(ib)+".npy", forces[ib])
  
#   print("forces.shape:", forces.shape)
#   print(forces[:, 0])
