import numpy as np
from mpi4py import MPI
from time import time

class Trajectory:
    def __init__(self, natom, comm=None):
        self.comm = comm
        if comm is not None:
          self.rank = comm.Get_rank()
          self.nbeads = comm.size
        else:
          self.rank = 0
          self.nbeads = 1
        self.natom = natom
        self.cells = None
        self.coords = None
        self.types = None
        self.vels = None
        self.forcenms = None
        self.forces = None
        self.images = None
        self.coords_unmap = None
        return None

    def read_npy(self):
        t0 = time()
        self.coords = np.load('coords'+str(self.rank)+'.npy')
        self.types = np.load('types'+str(self.rank)+'.npy')
        self.cells = np.load('cells'+str(self.rank)+'.npy')
        self.coords_unmap = np.load('coords_unmap'+str(self.rank)+'.npy')
        self.forcenms = np.load("forcenms"+str(self.rank)+".npy")
        t2 = time()
        if self.rank==0:
          print("Reading samples costs %.4f s."%(t2-t0))

    def xyz2npy(self, Ntraj):
        self.coords = np.zeros([Ntraj, natom, 3])
        self.types = np.zeros([Ntraj, natom])
        self.cells = np.zeros([Ntraj, 9])
        self.coords_unmap = np.zeros([Ntraj, natom, 3])
        self.images = np.zeros([Ntraj, natom, 3])
        self.vels = np.zeros([Ntraj, natom, 3])
        self.forcenms = np.zeros([Ntraj, natom, 3])

        t1 = time()
        with open("%02d"%(rank+1)+".xyz", "r") as f:
          for i in range(Ntraj):
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
              self.types[i][j] = int(line[1])
              self.coords[i][j] = np.array(line[2:5], dtype="float")
              self.vels[i][j] = np.array(line[5:8], dtype="float")
              self.images[i][j] = np.array(line[8:11], dtype="int")
              self.forcenms[i][j] = np.array(line[11:14], dtype="float")
            coords_unmap[i] = coords[i] + images[i]*np.array([xprd, yprd, zprd])
            if i%100 == 0:
              if rank == 0:
                t3 = time()
                print("Rank = %d: reading %d samples costs %.4f s.\n"%(rank, (i+1), t3-t1))
        t2 = time()
        print("Rank = %d: reading %d samples costs %.4f s.\n"%(rank, Ntraj, t2-t1))
        np.save("types"+str(rank)+".npy", types)
        np.save("coords"+str(rank)+".npy", coords)
        np.save("vels"+str(rank)+".npy", vels)
        np.save("coords_unmap"+str(rank)+".npy", coords_unmap)
        np.save("cells"+str(rank)+".npy", cells)