import glob
files = glob.glob("results/*trajectories*")
files.sort()
filename = files[-1]


import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(filename, delimiter=',')

#print(data[:,0])

frame_count = len(data[0])

HIST_BINS = 125


# TODO fix memory leak
for i in range(0, frame_count, 10):
    #gc.collect()
    fig, ax = plt.subplots(tight_layout=True, figsize=(10,4))
    ax.set_ylim((0,100))
    hist = ax.hist(data[:,i], range=[0,1], bins=HIST_BINS, density=True)
    ax.legend(["Probability of measuring |0>"])
    fig.savefig("plots/tmpframe{frame_number}.png".format(frame_number=i))
    #plt.show()
    print("Finished frame", i)
