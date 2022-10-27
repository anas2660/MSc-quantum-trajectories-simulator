import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob

curr_files = glob.glob("results/*currents*")
traj_files = glob.glob("results/*trajectories*")
curr_files.sort()
traj_files.sort()
curr_filename = curr_files[-1]
traj_filename = traj_files[-1]

data = np.loadtxt(curr_filename, delimiter=",")
probabilities = np.loadtxt(traj_filename, delimiter=",")[:, -1]

real = np.empty(1000)
imag = np.empty(1000)

for i in range(1000):
    real[i] = data[i, -2]
    imag[i] = data[i, -1]


colormap = cm.viridis(
    (probabilities - np.min(probabilities))
    / (np.max(probabilities) - np.min(probabilities))
)

plt.scatter(real, imag, c=colormap[:1000])
plt.colorbar(
    cm.ScalarMappable(
        norm=cm.colors.Normalize(
            vmin=np.min(probabilities), vmax=np.max(probabilities)
        ),
        cmap=cm.viridis,
    ),
    orientation="vertical",
    # label="Some Units",
)
plt.title("Integrated Current")
plt.xlabel("real")
plt.ylabel("imaginary")
plt.grid()
# plt.scatter(imag, t)
plt.show()


# plt.hist(real)
# plt.show()
#
# plt.hist(imag)
# plt.show()
