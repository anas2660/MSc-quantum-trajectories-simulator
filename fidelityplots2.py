#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("fidelity_data.csv", delimiter=",", skiprows=1)
index = data[:,0]
chi = data[:,1]
beta = data[:,2]
fidelity_mean = data[:,3]
fidelity_std = data[:,4]
indices = beta == 6.0
chi = chi[indices]
fidelity_mean = fidelity_mean[indices]
fidelity_std = fidelity_std[indices]

plt.rcParams["text.usetex"] = True

plt.figure(figsize=(6.5, 4))

plt.ylim(0, 1)

plt.grid()
plt.plot(chi, fidelity_mean)
plt.fill_between(chi, fidelity_mean-fidelity_std, fidelity_mean+fidelity_std, alpha=0.3)

plt.subplots_adjust(bottom=0.15)

plt.xlabel(r"\Large{$\chi$} \normalsize{[MHz]}")
plt.ylabel(r"\Large{Fidelity}")
plt.show()


fig, ax = plt.subplots(figsize=(6,6))
fidelity_mean = data[:,3]
new_len = int(np.sqrt(len(fidelity_mean)))
fidelity_mean = np.reshape(fidelity_mean, (new_len, new_len))
im = ax.imshow(fidelity_mean,interpolation='bicubic', extent=(0,4,20,0),vmin=0.0, vmax=1.0, aspect="auto")
fig.colorbar(im, ax=ax)
plt.xlabel("$\chi$")
plt.ylabel("$\\beta$")
plt.show()
