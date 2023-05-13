import glob
import numpy as np
import matplotlib.pyplot as plt


float_type = np.float32

files = glob.glob("results/*fidelity*")
files.sort()

step_files = glob.glob("results/*hist*")
step_files.sort()
sim_count_files = glob.glob("results/*current*")
sim_count_files.sort()
float_type_files = glob.glob("results/*trajectories*")
float_type_files.sort()

runs = len(files)

step_count = np.empty(runs)
simulations = np.empty(runs)
fidelity_mean = np.empty(runs)
fidelity_std = np.empty(runs)
chi = np.empty(runs)

print(runs)

for i in range(runs):

    step_buffer = open(step_files[i], "rb").read()
    sim_count_buffer = open(sim_count_files[i], "rb").read()
    float_type_buffer = open(float_type_files[i], "rb").read()
    data_buffer = open(files[i], "rb").read()

    float_type_number = np.frombuffer(float_type_buffer, np.uint32, count=4)[0]
    if float_type_number == 1:
        float_type = np.float64
    
    step_count[i] = np.frombuffer(step_buffer, np.uint32, count=3)[1]
    simulations[i] = np.frombuffer(sim_count_buffer, np.uint32, count=1)[0]

    data = np.reshape(np.frombuffer(data_buffer, float_type, offset=0), (int(simulations[i]), int(step_count[i]-1)))[:, -1]
    fidelity_mean[i] = np.mean(data)
    fidelity_std[i] = np.std(data)
    chi[i] = 0.01 * i


plt.rcParams["text.usetex"] = True

plt.figure(figsize=(6.5, 4))

plt.ylim(0, 1)

plt.plot(chi, fidelity_mean)
plt.fill_between(chi, fidelity_mean-fidelity_std, fidelity_mean+fidelity_std, alpha=0.3)

plt.subplots_adjust(bottom=0.15)

plt.xlabel(r"\Large{$\chi$} \normalsize{[MHz]}")
plt.ylabel(r"\Large{Fidelity}")


plt.show()

