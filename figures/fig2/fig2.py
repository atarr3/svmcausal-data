import matplotlib.pyplot as plt
import pandas as pd

# from matplotlib import rc
from matplotlib import rcParams
from os.path import abspath, dirname, join

# directories
WD = dirname(abspath(__file__))
RES_DIR = join(WD, *['..', '..', 'results'])
OUT_DIR = dirname(abspath(__file__))

# kernels and scenarios
kernels = ['linear', 'poly', 'rbf']
scenarios = ['G', 'cw']

# plot params
# rc('text', usetex=True)
# rc('text.latex', preamble=r'\usepackage{amsmath}')
rcParams["mathtext.fontset"] = 'cm'
constrained_layout = True # use tight layout if False

if __name__ == '__main__':
    ## Coverage Comparison ##

    fig, axs = plt.subplots(1, 3, figsize=(6.5, 4.333), sharey=True, 
                            constrained_layout=constrained_layout)
    if not constrained_layout: fig.subplots_adjust(wspace=0.1)
    
    for i in range(3):
        # read in data and get vectors
        res = pd.read_csv(join(RES_DIR, 'validation_%s.csv' % kernels[i]), 
                          index_col=0)
        
        C = res.index.to_numpy()
        coverage = res.coverage
        
        # scatter plot
        axs[i].scatter(C, coverage, label=r'SVM', s=15, alpha=0.3, zorder=10)
        
        # add y-label for left plot
        if i == 0: 
            axs[i].set_ylabel(r'Coverage', fontsize=12)
        
        # set axis
        axs[i].set_ylim(-0.05, 1.05)
        axs[i].set_xscale('log')
        axs[i].set_xlabel(r'$\lambda^{-1}$', fontsize=12)
        axs[i].tick_params(labelsize=10)
        axs[i].tick_params(axis='x', which='minor', bottom=False)
        if i == 0: axs[i].set_xticks([0.01, 0.1, 1])
        axs[i].set_aspect(1 / axs[i].get_data_ratio())    
        axs[i].grid(True)
    
    # layout and save
    if not constrained_layout: fig.tight_layout()
    plt.savefig(join(OUT_DIR,'fig2.pdf'), dpi=200, bbox_inches='tight')
