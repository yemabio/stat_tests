import math
import numpy as np
import pandas as pd
import scipy.stats.distributions as dist
import matplotlib.pyplot as plt

def compute_required_sample_size(delta, stdev, power=0.8, significance=0.05):
    '''
    Computes power of Wald test
    '''
    z_half_alpha = np.abs(dist.norm.ppf(significance/2))
    z_beta = dist.norm.ppf(power)

    n = (2*(z_half_alpha+z_beta)**2*stdev**2)/delta**2

    return n

def plot_sample_size_versus_power(delta, stdev, significance=0.05,power_range=(0,1),critical_power=0.8):
    powers = np.linspace(power_range[0],power_range[1])
    n_range = [compute_required_sample_size(delta, stdev,power=p,significance=significance) for p in powers]

    critical_n = compute_required_sample_size(delta, stdev,power=critical_power,significance=significance)

    fig, ax = plt.subplots(1,1)
    ax.plot(powers,n_range,color='k')
    ax.vlines(0.8, ymin = 0, ymax = critical_n, color = 'blue', linestyle = '--', label=rf'critical $n = $ {np.round(critical_n,2)}')
    ax.hlines(critical_n, xmin=0, xmax=critical_power, color='blue',linestyle='--')
    ax.legend()
    ax.set_ylabel(r'Required $n$')
    ax.set_xlabel('Desired power')
    return fig,ax
    




