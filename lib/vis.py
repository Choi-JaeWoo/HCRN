from scipy import cluster
import matplotlib.pyplot as plt
import numpy as np

def plot_rho_perf(rho_list, mean_dict, std_dict, save_path='./result.jpg'):
    algorithm_list = list(mean_dict.keys())
    fig, ax = plt.subplots()
    for al in algorithm_list:
        ax.errorbar(rho_list, mean_dict[al], yerr=np.asarray(std_dict[al]), marker='o', ms=5, label=al)
    ax.legend()
    ax.grid()
    plt.savefig(save_path)
    plt.close(fig)

