import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob

files = glob.glob("results/*currents*")
files.sort()
filename = files[-1]
simulations = np.fromfile(filename, np.int32)[0]
steps = np.fromfile(filename, np.int32)[1]

data = np.fromfile(filename, np.float32)[2:]
data = np.reshape(data, (simulations, steps, 2))


files = glob.glob("results/*trajectories*")
files.sort()
filename = files[-1]
probabilities = np.fromfile(filename, np.float32)[2:]
probabilities = np.reshape(probabilities, (simulations, steps + 1))[:, -1]
print(np.size(probabilities))


real = data[:, -1, 0]
imag = data[:, -1, 1]


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
    label="probabilities",
)
plt.title("Integrated Current")
plt.xlabel("real")
plt.ylabel("imaginary")
plt.grid()
plt.show()
