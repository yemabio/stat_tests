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


    # upper = dist.norm.cdf((theta_0-theta_star)/sig + z_half_alpha)
    # lower = dist.norm.cdf((theta_0-theta_star)/sig - z_half_alpha)
    # return 1-upper+lower

def plot_sample_size_versus_power(delta, stdev, significance=0.05,power_range=(0,1)):
    powers = np.linspace(power_range[0],power_range[1])
    n_range = [compute_required_sample_size(delta, stdev,power=p,significance=significance) for p in powers]
    fig, ax = plt.subplots(1,1)
    ax.plot(powers,n_range,color='k')
    # ax.vlines(0.8, ymin = 0, ymax = 1, color = 'blue', linestyle = '-', transform=ax.get_xaxis_transform())
    # ax.hlines(0.8, ymin = 0, ymax = 1, color = 'blue', linestyle = '-', transform=ax.get_xaxis_transform())
    ax.set_ylabel(r'Required $n$')
    ax.set_xlabel('Desired power')
    return fig,ax
    




