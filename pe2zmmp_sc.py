import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help="task id")
parser.add_argument("--nbead", type=int, help="the number of beads")
parser.add_argument("--nsamp", type=int, help="the number of samples")
args = parser.parse_args()
ndiscard = 0000

nbead = args.nbead
nsamp = args.nsamp
pe = np.zeros([nbead, nsamp])
pe_Osc = np.zeros([nbead, nsamp])
pe_Hsc = np.zeros([nbead, nsamp])
directory = "../fractionation/task"+str(args.id)+"/"

temp = 300.0
kB = 8.617343e-5
hplanck = 4.135667403e-3
mvv2e = 1.0364269e-4
hbar = hplanck / 2 / np.pi
beta = 1.0 / (kB * temp)
beta_np = 1.0 / (kB * temp) / nbead
omega_np = nbead / (hbar * beta) * mvv2e**0.5

for i in range(nbead):
    pe[i] = np.loadtxt(directory+"pe.%02d"%(i+1))[ndiscard:ndiscard+nsamp, 1]
    pe_Osc[i] = np.loadtxt(directory+"pe_Osc.%02d"%(i+1))[ndiscard:ndiscard+nsamp, 1]
    pe_Hsc[i] = np.loadtxt(directory+"pe_Hsc.%02d"%(i+1))[ndiscard:ndiscard+nsamp, 1]
    
pe_sum = pe.sum(axis=0)
pe_Osc_sum = pe_Osc.sum(axis=0)
pe_Hsc_sum = pe_Hsc.sum(axis=0)

zmmp_sc_O = np.exp(-beta_np*(pe_Osc_sum - pe_sum))
zmmp_sc_H = np.exp(-beta_np*(pe_Hsc_sum - pe_sum))

np.save(directory+"zmmp_sc_O.npy", zmmp_sc_O)
np.save(directory+"zmmp_sc_H.npy", zmmp_sc_H)