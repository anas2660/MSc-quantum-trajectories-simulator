#!/usr/bin/python
import glob
files = glob.glob("results/*trajectories*")
files.sort()
filename = files[-1]


import numpy as np
import matplotlib.pyplot as plt


float_type = np.float64




data_buffer = open(filename, "rb").read()
data = np.frombuffer(data_buffer, float_type, offset=12)
metadata = np.frombuffer(data_buffer, np.uint32, count=3)

simulation_count = metadata[0]
state_count = metadata[1]
step_count = metadata[2]

t = data[step_count*state_count:]
data = np.reshape(data[0:step_count*state_count], (step_count, state_count))

#errors = np.std(data, axis=0)
# data = np.average(data, axis=0)

FORMAT_STRING = '|{0:0' + str(int(np.log2(state_count))) + 'b}>'

for i in np.arange(state_count):
    plt.plot(t, data[:, i], linewidth=3, label=FORMAT_STRING.format(i))

# plt.plot(np.sum(data, axis=1), linewidth=3)
plt.legend()
plt.grid()
plt.xticks(np.arange(np.ceil(t.max())+1))
plt.show()


plt.imshow(np.swapaxes(data, 0, 1), origin='upper', extent=[0, t.max(), float(state_count), 0], aspect='auto')
plt.xlabel("Time")
plt.ylabel("States")
plt.yticks(ticks=(np.arange(state_count)+0.5),
           labels = [FORMAT_STRING.format(i) for i in np.arange(state_count)])
plt.grid()
plt.colorbar()
plt.show()


# data = np.loadtxt(filename, delimiter=',')

#print(data[:,0])

frame_count = len(data[0])

HIST_BINS = 125


files = glob.glob("results/*hist*")
files.sort()
filename = files[-1]

hist_data_buffer = open(filename, "rb").read()
hist_metadata = np.frombuffer(hist_data_buffer, np.uint32, count=3)
hist_state_count = hist_metadata[0]
hist_step_count = hist_metadata[1]
hist_height = hist_metadata[2]
hist_data = np.reshape(np.frombuffer(hist_data_buffer, np.uint32, offset=12), (hist_step_count, hist_height))

from matplotlib import colors
from matplotlib import pyplot as plt, cm

fig, ax = plt.subplots()
print(np.max(hist_data))
im = ax.imshow(np.swapaxes(np.minimum(hist_data + np.min(hist_data[hist_data>0]), 10000000), 0, 1), origin='upper', extent=[0, t.max(), float(hist_state_count), 0], aspect='auto',
               cmap=cm.turbo,
               norm=colors.LogNorm()
               )
ax.set_xlabel("Time")
ax.set_ylabel("States")
ax.set_yticks(ticks=(np.arange(hist_state_count)), minor=True)
ax.set_yticks(ticks=(np.arange(hist_state_count)+0.5),
           labels = [FORMAT_STRING.format(i) for i in np.arange(hist_state_count)])
ax.grid(True, which='minor', linewidth=2.0)
ax.grid(False, which='major')
fig.colorbar(im,
             norm=colors.LogNorm()
             )
plt.tight_layout()
plt.show()




files = glob.glob("results/*current*")
files.sort()
filename = files[-1]


current_data_buffer = open(filename, "rb").read()
current_metadata = np.frombuffer(current_data_buffer, np.uint32, count=1)
current_simulations = current_metadata[0]
current_data = np.reshape(np.frombuffer(current_data_buffer, float_type, offset=4), (current_simulations, 2))

plt.plot(current_data[:,0], current_data[:,1], 'o')

avg_x = np.average(current_data[:,0])
avg_y = np.average(current_data[:,1])
plt.plot(avg_x, avg_y, 'ro')
plt.grid()


print(avg_x, ",", avg_y)

plt.show()



# TODO fix memory leak
if False:
    for i in range(0, frame_count, 1):
        #gc.collect()
        fig, ax = plt.subplots(tight_layout=True, figsize=(10,4))
        ax.set_ylim((0,25))
        hist = ax.hist(data[:,i], range=[0,1], bins=HIST_BINS, density=True)
        ax.legend(["Probability of measuring |0>"])
        fig.savefig("plots/tmpframe{frame_number}.png".format(frame_number=i))
        #plt.show()
        print("Finished frame", i)
