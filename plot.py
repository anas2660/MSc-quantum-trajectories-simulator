import glob
files = glob.glob("results/*trajectories*")
files.sort()
filename = files[-1]


import numpy as np
import matplotlib.pyplot as plt

data_buffer = open(filename, "rb").read()
data = np.frombuffer(data_buffer, np.float32, offset=12)
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

print(state_count)


plt.imshow(np.swapaxes(data, 0, 1), origin='upper', extent=[0, t.max(), float(state_count), 0], aspect='auto')
plt.xlabel("Time")
plt.ylabel("States")
plt.yticks(ticks=(np.arange(state_count)+0.5),
           labels = [FORMAT_STRING.format(i) for i in np.arange(state_count)]
           )
plt.grid()
plt.show()


# data = np.loadtxt(filename, delimiter=',')

#print(data[:,0])

frame_count = len(data[0])

HIST_BINS = 125



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
