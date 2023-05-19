#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np


data = np.loadtxt("purity_data.csv", delimiter=',', skiprows=1)

t = data[:,0]
μ = data[:,1]
s = data[:,2]

plt.plot(t,μ)
plt.fill_between(t, μ-s, μ+s, alpha=0.3)
plt.ylim(0,1)
plt.show()


data = np.loadtxt("purity_data_varying_chi.csv", delimiter=',', skiprows=1)

Δχ = data[:,0]
μ = data[:,1]
s = data[:,2]

plt.plot(Δχ, μ)
plt.fill_between(Δχ, μ-s, μ+s, alpha=0.3)
plt.ylim(0,1)
plt.show()
