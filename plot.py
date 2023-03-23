#!/usr/bin/python
import glob
files = glob.glob("results/*trajectories*")
files.sort()
filename = files[-1]


import numpy as np
import matplotlib.pyplot as plt


float_type = np.float32




data_buffer = open(filename, "rb").read()
metadata = np.frombuffer(data_buffer, np.uint32, count=4)

float_type_number = metadata[0]
print("Found float precision: ", float_type_number)
if float_type_number == 1:
    float_type = np.float64

data = np.frombuffer(data_buffer, float_type, offset=16)
print(len(data))

simulation_count = metadata[1]
state_count = metadata[2]
step_count = metadata[3]

t = data[step_count*state_count:]
data = np.reshape(data[0:step_count*state_count], (step_count, state_count))

#errors = np.std(data, axis=0)
# data = np.average(data, axis=0)

FORMAT_STRING = '|{0:0' + str(int(np.log2(state_count))) + 'b}>'

for i in np.arange(state_count):
    plt.plot(t, data[:, i], linewidth=3, label=FORMAT_STRING.format(i))

plt.plot(t, np.sum(data, axis=1), linewidth=3)
plt.legend()
plt.grid()
#plt.xlim(t.max())
#plt.xticks(np.linspace(0,t.max(), 10))
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


# Current plot data
files = glob.glob("results/*current*")
files.sort()
filename = files[-1]

current_data_buffer = open(filename, "rb").read()
current_metadata = np.frombuffer(current_data_buffer, np.uint32, count=1)
current_simulations = current_metadata[0]
current_data = np.reshape(np.frombuffer(current_data_buffer, float_type, offset=4), (current_simulations, 2))

files = glob.glob("results/*final_state*")
files.sort()
filename = files[-1]
final_state_data_buffer = open(filename, "rb").read()
final_state_data = np.reshape(np.frombuffer(final_state_data_buffer, float_type, offset=0), (current_simulations, state_count))

final_state_colors = final_state_data[:,:3]
final_state_colors = final_state_data[:,0]

# Current plot actual plotting
from matplotlib.widgets import Slider
initial_state = int(0)
fig, ax = plt.subplots()
fig.subplots_adjust(left=0.1, bottom=0.15, right=0.99, top=0.95)

points = ax.scatter(
    current_data[:,0], current_data[:,1], 
    c=final_state_colors, cmap=cm.turbo,
    vmin=0.0, vmax = 1.0)

avg_x = np.average(current_data[:,0])
avg_y = np.average(current_data[:,1])
#plt.plot(avg_x, avg_y, 'ro')
ax.grid()
print(avg_x, ",", avg_y)

# make slider for selecting state
axslider = fig.add_axes([0.05, 0.065, 0.9, 0.03])
slider = Slider(
    ax=axslider,
    label="",
    valmin=0,
    valmax=state_count-1,
    valinit=initial_state,
    valstep=1,
    valfmt='%0.0f'
)

axslider.add_artist(axslider.xaxis)
axslider.set_xticks(ticks = np.arange(0.0, state_count, dtype=int), labels = [FORMAT_STRING.format(i) for i in np.arange(0.0, state_count, dtype=int)])

fig.colorbar(points)

def update(val):
    points.set_array(final_state_data[:, slider.val])
    fig.canvas.draw_idle()
slider.on_changed(update)

plt.show()



# Current plot actual plotting stick thing
initial_state = int(0)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
fig.subplots_adjust(left=0.1, bottom=0.15, right=0.99, top=0.95)

#points = ax.scatter(current_data[:,0], current_data[:,1], final_state_data[:, 0])
markerline, stemlines, baseline = ax.stem(current_data[:,0], current_data[:,1], final_state_data[:, 0], basefmt=" ")
#, linefmt='grey'

ax.set(xlabel='i', ylabel='q', zlabel='probability')
ax.set_zlim(0,1)
ax.grid()
print(avg_x, ",", avg_y)

# make slider for selecting state
axslider = fig.add_axes([0.05, 0.065, 0.9, 0.03])
slider = Slider(
    ax=axslider,
    label="",
    valmin=0,
    valmax=state_count-1,
    valinit=initial_state,
    valstep=1,
    valfmt='%0.0f'
)
axslider.add_artist(axslider.xaxis)
axslider.set_xticks(ticks = np.arange(0.0, state_count, dtype=int), labels = [FORMAT_STRING.format(i) for i in np.arange(0.0, state_count, dtype=int)])

def update2(val):
    #points.set_array(final_state_data[:, slider.val])
    ax.cla()
    markerline, stemlines, baseline = ax.stem(current_data[:,0], current_data[:,1], final_state_data[:, slider.val], basefmt=" ")
    ax.set_zlim(0,1)
    #plt.setp(markerline, marker='D', markersize=10, markeredgecolor="orange", markeredgewidth=2)
    fig.canvas.draw_idle()
slider.on_changed(update2)

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
