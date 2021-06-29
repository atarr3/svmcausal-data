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
    ## Objective comparison ##
    
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 4.333), sharey=False, 
                            constrained_layout=constrained_layout)
    if not constrained_layout: fig.subplots_adjust(wspace=0.1)
    
    for i in range(3):
        # read in data and get vectors
        res = pd.read_csv(join(RES_DIR, 'validation_%s.csv' % kernels[i]), 
                          index_col=0)
        
        C = res.index.to_numpy()
        obj_svm = res.obj_svm
        obj_qip = res.obj_qip
        
        # plot curves
        axs[i].plot(C, obj_svm, label=r'SVM', lw=1.5, alpha=1, zorder=2)
        axs[i].plot(C, obj_qip, label=r'SVM-QIP', lw=1.5, ls='--', alpha=1, zorder=1)
        
        # add legend and y-label for left plot
        if i == 0: 
            axs[i].set_ylabel(r'Objective', fontsize=12)
            axs[i].legend(framealpha=1, prop={"size": 8})
        
        # set axis
        axs[i].set_xscale('log')
        axs[i].set_xlabel(r'$\lambda^{-1}$', fontsize=12)
        axs[i].tick_params(labelsize=10)
        axs[i].tick_params(axis='x', which='minor', bottom=False)
        if i == 0: axs[i].set_xticks([0.01, 0.1, 1])
        axs[i].set_aspect(1 / axs[i].get_data_ratio())    
        axs[i].grid(True)
    
    # layout and save
    if not constrained_layout: fig.tight_layout()
    plt.savefig(join(OUT_DIR,'fig1.pdf'), dpi=200, bbox_inches='tight')