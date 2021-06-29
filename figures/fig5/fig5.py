import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from matplotlib import rc
from matplotlib import rcParams
from os.path import abspath, dirname, join

# directories
WD = dirname(abspath(__file__))
RES_DIR = join(WD, *['..', '..', 'results'])
OUT_DIR = dirname(abspath(__file__))

# results
SVM_PATH = join(RES_DIR, 'rhc_svm.csv')

# kernels and scenarios
kernels = ['linear', 'poly', 'rbf']
scenarios = ['G', 'cw']

# plot params
# rc('text', usetex=True)
# rc('text.latex', preamble=r'\usepackage{amsmath}')
rcParams["mathtext.fontset"] = 'cm'
constrained_layout = True # use tight layout if False

if __name__ == '__main__':
    # SVM data
    svm = pd.read_csv(SVM_PATH, index_col=[0,1,2])
    
    ## ATE vs. Balance plots ##
    
    # plot settings
    ylims = [-0.01, 0.125]
    yticks = [0.00, 0.04, 0.08, 0.12]
    
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 4.333), sharey=True, 
                            constrained_layout=constrained_layout)
    if not constrained_layout: fig.subplots_adjust(wspace=0.1)
    
    for i in range(3):
        kernel = kernels[i]
        sub = svm.xs([kernel], level=['kernel'])
        
        # C values
        C = np.unique(sub.index)
        
        # relevant values
        ate = sub.ate
        se = sub.se
        bal = sub.bal
        
        # plot curves
        axs[i].plot(bal, ate, label=r'SVM', alpha=1, lw=1.5, zorder=12)
        axs[i].fill_between(bal, ate - 1.96 * se, ate + 1.96 * se, color='gray', 
                            alpha=0.25, zorder = 8)
        
        # axis
        if i == 1: axs[i].set_xlabel(r'Normed difference-in-means', fontsize=12)
        if i == 0: axs[i].set_ylabel(r'ATE', fontsize=12)
        axs[i].set_ylim(ylims)
        axs[i].set_yticks(yticks)
        axs[i].tick_params(labelsize=10)
        axs[i].set_aspect(1 / axs[i].get_data_ratio())    
        axs[i].grid(True)
        
    # layout and save
    fig.align_ylabels()
    if not constrained_layout: fig.tight_layout()
    plt.savefig(join(OUT_DIR,'fig5.pdf'), dpi=200, bbox_inches='tight')