import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob

files = glob.glob("results/*currents*")
files.sort()
filename = files[-1]

data_buffer = open(filename, "rb").read()
data = np.frombuffer(data_buffer, np.float32, offset=8)
metadata = np.frombuffer(data_buffer, np.uint32, count=2)

data = np.reshape(data, (metadata[0], metadata[1], 2))

files = glob.glob("results/*trajectories*")
files.sort()
filename = files[-1]

probabilities = np.fromfile(filename, np.float32, offset=8)
probabilities = np.reshape(probabilities, (metadata[0], metadata[1] + 1))[:, -1]

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
    label="Probabilities",
)
plt.title("Integrated Current")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.grid()
plt.show()
