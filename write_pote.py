import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help="task id")
parser.add_argument("--nbead", type=int, help="the number of beads")
parser.add_argument("--start", type=int, help="start line number")
parser.add_argument("--end", type=int, help="end line number")
parser.add_argument("--pot", type=int, help="potential energy column number")
args = parser.parse_args()

nbead = args.nbead
start = args.start
end = args.end
pot = args.pot
directory = "../h2o/task"+str(args.id)+"/"
for i in range(nbead):
    with open(directory+"log."+str(i), "r") as f:
        lines = []
        for j in range(start-1):
            f.readline()
        for j in range(end-start+1):
            lines.append(f.readline().split())
        lines = np.array(lines, dtype="float")
        np.save(directory+"pote"+str(i)+".npy", lines[:, pot-1])