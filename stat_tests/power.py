import math
import numpy as np
import pandas as pd
import scipy.stats.distributions as dist
from scipy.stats import t as t_dist
from scipy.stats import nct
import matplotlib.pyplot as plt



# def compute_required_sample_size_within(delta, std, power=0.8,significance=0.05,n_measurements=2,correlation=0.5):
#     '''
#     from https://aaroncaldwell.us/SuperpowerBook/repeated-measures-anova.html and Faul,Erdfelder,Lang,and Buchner
#       https://courses.washington.edu/psy524a/_book/two-sample-independent-measures-t-test.html
#     '''
#     effect_size = delta/std

def compute_power_t(mu1, mu2, stdev1, stdev2, n1, n2=None, significance=0.05, sample_corr=0, two_sided=True):
    #from IBM SPSS Statistics Algorithms
    # if stdev1 == stdev2:
    if sample_corr:
        dof = n1-1
        non_cent_den = np.sqrt(stdev1**2+stdev2**2-2*sample_corr*stdev1*stdev2)/np.sqrt(n1)
    else:
        dof = n1+n2-2
        non_cent_den = np.sqrt(stdev1**2/n1+stdev2**2/n2)
    # else:
        # dof_num = (stdev1**2/n1 + stdev2**2/n2)**2
        # dof_den = (stdev1**2/n1)**2/(n1-1) + (stdev2**2/n2)**2/(n2-1)
        # dof = dof_num/dof_den

    if two_sided:
        t_crit = t_dist.ppf(1-significance/2,dof)
        non_cent = (mu1-mu2)/non_cent_den
        power = 1-nct.cdf(t_crit, dof, non_cent) + nct.cdf(-t_crit, dof, non_cent)

    else: 
        t_crit = t_dist.ppf(1-significance,dof)
        non_cent = np.abs(mu1-mu2)/non_cent_den
        power = 1-nct.cdf(t_crit, dof, non_cent)

    return(power)

def plot_sample_size_versus_power_t(mu1, mu2, stdev1, stdev2, significance=0.05, sample_corr=0, two_sided=True, critical_power=0.8, n_range=(1,30)):
    fig, ax = plt.subplots(1,1)
    ns = np.linspace(n_range[0],n_range[1],num=200)
    powers = np.array([compute_power_t(mu1,
                                       mu2,stdev1,
                                       stdev2,
                                       n,
                                       n,
                                       significance=significance,
                                       sample_corr=sample_corr,
                                       two_sided=two_sided) for n in ns])
    try:
        critical_n = ns[np.where(powers > critical_power)[0][0]]
    except IndexError as _:
        print('Critical n not met. Try increasing n_range upper bound.')
        return fig, ax


    # fig, ax = plt.s
    # print(powers)
    ax.plot(powers,ns,color='k')
    ax.vlines(0.8, ymin = 0, ymax = critical_n, color = 'blue', linestyle = '--', label=rf'critical $n = $ {np.round(critical_n,2)} per treatment')
    ax.hlines(critical_n, xmin=0, xmax=critical_power, color='blue',linestyle='--')
    ax.legend()
    ax.set_xlim((0,1))
    ax.set_ylim(n_range)
    ax.set_ylabel(r'Required $n$')
    ax.set_xlabel('Desired power (t-test)')
    return fig, ax

def plot_sample_size_versus_power_z(delta, stdev, stdev2=None, significance=0.05,power_range=(0,1),critical_power=0.8):
    powers = np.linspace(power_range[0],power_range[1])
    n_range = [compute_required_sample_size_z(delta, stdev,stdev2=stdev2, power=p,significance=significance) for p in powers]

    critical_n = compute_required_sample_size_z(delta, stdev,stdev2=stdev2, power=critical_power,significance=significance)

    fig, ax = plt.subplots(1,1)
    ax.plot(powers,n_range,color='k')
    ax.vlines(0.8, ymin = 0, ymax = critical_n, color = 'blue', linestyle = '--', label=rf'critical $n = $ {np.round(critical_n,2)}')
    ax.hlines(critical_n, xmin=0, xmax=critical_power, color='blue',linestyle='--')
    ax.legend()
    ax.set_ylabel(r'Required $n$')
    ax.set_xlabel('Desired power (z-test)')
    return fig,ax

def compute_required_sample_size_z(delta, stdev, stdev2=None, power=0.8, significance=0.05):
    '''
    Computes power of Wald test
    '''
    z_half_alpha = np.abs(dist.norm.ppf(significance/2))
    z_beta = dist.norm.ppf(power)

    if stdev2:
        n = ((z_half_alpha+z_beta)**2*(stdev**2+stdev2**2))/delta**2
    else:
        n = ((z_half_alpha+z_beta)**2*2*stdev**2)/delta**2
    return n
    




