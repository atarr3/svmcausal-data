import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from itertools import product
# from matplotlib import rc
from matplotlib import rcParams
from os.path import abspath, dirname, exists, join

# directories
WD = dirname(abspath(__file__))
RES_DIR = join(WD, *['..', '..', 'results'])
OUT_DIR = dirname(abspath(__file__))

# results
SVM_PATH = join(RES_DIR, 'sims_svm.csv')

def combine_results(data='sims', method='svm'):
    if data not in ['sims', 'rhc']: return -1
    if method not in ['svm', 'kom']: return -1
    
    kernels = ['linear', 'poly', 'rbf']
    scenarios = ['G', 'cw'] if data == 'sims' else ['rhc']
    
    df = pd.DataFrame()
    
    for scenario in scenarios:
        for kernel in kernels:
            # read in data
            if scenario == 'rhc':
                fname = '%s_%s_%s.csv' % (scenario, method, kernel)
                svm = pd.read_csv(join(RES_DIR, fname),
                                  index_col=[0]
                                  )
                svm.index.rename('C', inplace=True)
            else:
                fname = '%s_%s_%s.csv' % (scenario, method, kernel)
                svm = pd.read_csv(join(RES_DIR, fname),
                                  index_col=[0,1])
            # reindex
            svm['kernel'] = kernel
            svm['scenario'] = scenario
            temp = svm.set_index([svm.index, svm.kernel, svm.scenario]
                                ).drop(columns=['scenario', 'kernel'])
            
            # update
            df = df.append(temp)
            
    # save
    df.to_csv(join(RES_DIR, '%s_%s.csv' % (data, method)))
    
    return df

if __name__ == '__main__':
    # combine data if no single file for SVM results found
    if not exists(SVM_PATH):
        try:
            combine_results('sims')
        except FileNotFoundError:
            print('missing results, please run simulations.py')
        
    # SVM data
    svm = pd.read_csv(SVM_PATH, index_col=[0,1,2,3])
    
    # kernels and scenarios
    kernels = ['linear', 'poly', 'rbf']
    scenarios = ['G', 'cw']
    
    # plot params
    # rc('text', usetex=True)
    # rc('text.latex', preamble=r'\usepackage{amsmath}')
    rcParams["mathtext.fontset"] = 'cm'
    constrained_layout = True # use tight layout if False
    
    # hatches
    hatches = [None, '///', '...']
    
    # index slicer
    idx = pd.IndexSlice
    
    # width transformer
    width = lambda p, w: 10 ** (np.log10(p) + w/2.) - 10 ** (np.log10(p) - w/2.)
    
    ## Path Plots ##
    
    fig, axs = plt.subplots(2, 3, figsize=(6.5, 5), sharey='row',
                            constrained_layout=constrained_layout)
    if not constrained_layout: fig.subplots_adjust(wspace=0.1, hspace=0.2)
    
    for (i,j) in product(range(2),range(3)):
        # read in data and compute statistics
        scenario = scenarios[i]
        kernel = kernels[j]
        hatch = hatches[j]
        res = svm.xs([kernel, scenario], level=['kernel', 'scenario'])
        ate = res.groupby(level='C').ate.mean()
        se = res.groupby(level='C').se.mean()
        sd = res.groupby(level='C').ate.std()
        
        # C values
        C = np.unique(res.index.droplevel(0))
        # true PATE
        tau = 10 if scenario == 'cw' else -0.4
        
        # get small sample of points
        C_sub = C[::12] # from `markevery = 12` parameter in plot
        res_sub = res.loc[idx[:, C_sub], :]
        
        # true effect
        axs[i,j].axhline(y=tau, linestyle='--', color='#d62728', 
                        alpha=1, lw=1, zorder=9)
        
        bp_svm = axs[i,j].boxplot(np.array(res_sub.groupby('C').ate.apply(list).tolist(), 
                                           dtype=object).T,
                              positions=C_sub,
                              medianprops={'color' : 'black', 'lw' : 2},
                              widths=width(C_sub, 0.2),
                              patch_artist=True, showfliers=False, zorder=10)
            
        for patch in bp_svm['boxes']:
            patch.set_facecolor("white")
            patch.set(hatch=hatch)
        
        # axis labels      
        axs[i,j].set_xlabel(r'$\lambda^{-1}$', fontsize=12)
        # y-axis label on left plots only
        if j == 0:
            axs[i,j].set_ylabel(r'ATE', fontsize=12)
        # title on middle plots only
        sim = 'A' if scenario == 'G' else 'B'
        if j == 1: 
            axs[i,j].set_title(r'Simulation %s' % sim, fontsize=12)
        # axis formatting
        axs[i,j].set_xscale('log')
        if scenario != 'cw':
            axs[i,j].set_ylim(-0.62, -0.08)
        else:
            axs[i,j].set_ylim(-6, 21)
        axs[i,j].tick_params(labelsize=10)
        axs[i,j].set_aspect(1 / axs[i,j].get_data_ratio())    
        axs[i,j].grid(True)
    
    # layout and save
    fig.align_ylabels()
    if not constrained_layout: fig.tight_layout()
    plt.savefig(join(OUT_DIR,'fig3.pdf'), dpi=200, bbox_inches='tight')
