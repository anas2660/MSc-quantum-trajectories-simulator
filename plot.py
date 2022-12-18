import glob
files = glob.glob("results/*trajectories*")
files.sort()
filename = files[-1]


import numpy as np
import matplotlib.pyplot as plt

data_buffer = open(filename, "rb").read()
data = np.frombuffer(data_buffer, np.float32, offset=12)
metadata = np.frombuffer(data_buffer, np.uint32, count=3)

data = np.reshape(data, (metadata[0], metadata[2], metadata[1]))

errors = np.std(data, axis=0)
data = np.average(data, axis=0)
plt.plot(data[:, 0], linewidth=3, label='00')
plt.plot(data[:, 1], linewidth=3, label='01')
plt.plot(data[:, 2], linewidth=3, label='10')
plt.plot(data[:, 3], linewidth=3, label='11')
# plt.plot(np.sum(data, axis=1), linewidth=3)
plt.legend()
plt.grid()
plt.show()


plt.imshow(np.swapaxes(data, 0, 1), origin='upper', extent=[-1, 1, -1, 1], aspect='auto')
plt.xlabel("Time")
plt.ylabel("States")
plt.yticks(ticks=(np.array([3,2,1,0])-1.0)/2.0 - 0.25, labels=['|00>','|01>','|10>','|11>'])
plt.grid()
plt.show()


# data = np.loadtxt(filename, delimiter=',')

#print(data[:,0])

frame_count = len(data[0])

HIST_BINS = 125



# TODO fix memory leak
if 0:
    for i in range(0, frame_count, 1):
        #gc.collect()
        fig, ax = plt.subplots(tight_layout=True, figsize=(10,4))
        ax.set_ylim((0,25))
        hist = ax.hist(data[:,i], range=[0,1], bins=HIST_BINS, density=True)
        ax.legend(["Probability of measuring |0>"])
        fig.savefig("plots/tmpframe{frame_number}.png".format(frame_number=i))
        #plt.show()
        print("Finished frame", i)
