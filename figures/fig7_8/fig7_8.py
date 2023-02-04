import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as npm
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

# function for computing elbow point of a curve y = f(x)
def elbow(x, y):
    # sort x to be in ascending order
    xs = np.sort(x.to_numpy())
    ys = y.to_numpy()[np.argsort(x)]
    
    # form tuples of x,y
    coords = np.vstack((xs,ys)).T
    # initial/final points
    p0 = coords[0]
    p1 = coords[-1]
    
    # normalized line joining points
    vec = p1 - p0
    vec = vec / np.sqrt(np.sum(vec ** 2))
    
    # distances from vec
    vec_p0 = coords - p0
    ips = np.sum(vec_p0 * np.matlib.repmat(vec, len(xs), 1), axis=1)
    points = np.outer(ips, vec)
    # vectors from points to line, perpendicular to vec
    vecs = vec_p0 - points
    dists_vec = np.sum(vecs ** 2, axis=1)
    
    return np.argsort(x).to_numpy()[dists_vec.argmax()]

if __name__ == '__main__': 
    # SVM data
    svm = pd.read_csv(SVM_PATH, index_col=[0,1,2])
    
    ## SDIM Comparisons ##

    # arrays
    methods = ['card', 'kom']
    kernels = ['linear', 'poly']
    scenarios = ['G', 'cw']
    fnums = [7, 8]
    
    # plot params
    # rc('text', usetex=True)
    # rc('text.latex', preamble=r'\usepackage{amsmath}')
    rcParams["mathtext.fontset"] = 'cm'
    constrained_layout = True # use tight layout if False
    
    # plot params
    alpha = 0.3
    
    for m, method in enumerate(methods):
        # number of comparisons
        ncomps = 4
        
        # plot parameters
        if method == 'card':
            # axis range and tick marks for sdim
            ax_range = [-0.01, 0.26]
            ax_ticks = [0.00, 0.10, 0.20]
        
        # create figure
        if method == 'card':
            fig, axs = plt.subplots(2, 4, figsize=(6.5, 3.75), sharey=True, 
                                    constrained_layout=constrained_layout)
        else:
            fig, axs = plt.subplots(1, 2, figsize=(6.5, 1.875), sharey=False, 
                                    constrained_layout=constrained_layout)
        if not constrained_layout: fig.subplots_adjust(wspace=0.1)
        
        for i, kernel in enumerate(kernels):
            # read in data
            res_oth = pd.read_csv(join(RES_DIR, 
                                       'rhc_%s_%s.csv' % (method, kernel)
                                      )
                                 )
            # get SVM data (drop NaN columns)
            res_svm = svm.xs(kernel, level='kernel').dropna(axis=1)
            
            # get SDIM
            # sdim results for each C
            sdim_svm = res_svm.drop(
                ['ate','se', 'bal', 'bal_sd', 'ss', 'ess', 'asum'], axis=1)
            # sdim results for other method
            sdim_oth = res_oth.drop(
                ['ate','se', 'bal', 'bal_sd', 'ss', 'ess'], axis=1).iloc[0]
            
            # subset size for other method
            ss_oth = res_oth.ess[0]
            
            # card plots
            if method == 'card':
                # loop through comparisons
                for j in range(ncomps):
                    # sdim <= 0.1
                    if j == 0:
                        ind = np.where((np.abs(sdim_svm) <= 0.1).all(axis=1))[0][0]
                        
                    # closest ess
                    if j == 1:
                        ind = len(res_svm) - np.abs(res_svm.ess - ss_oth)[::-1].argmin() - 1
                        
                    # equal ndim
                    if j == 2:
                        ndim_oth = res_oth.bal[0]                   
                        ind = np.where(res_svm.bal <= ndim_oth)[0][0]
                                        
                    # elbow solution
                    if j == 3:
                        if kernel == 'linear':
                            ind = -1
                        else:
                            ind = elbow(res_svm.asum, res_svm.bal)
                            
                    # get corresponding SVM solution
                    sdim_sub = sdim_svm.iloc[ind]
                    ss = res_svm.iloc[ind].ess
                            
                    # plot
                    axs[i,j].scatter(np.abs(sdim_sub), np.abs(sdim_oth), 
                                     s=10, alpha=alpha, zorder=12)
                    
                    # y = x line
                    lims = ax_range
                    axs[i,j].plot(lims, lims, 'k--', lw=1, zorder=0)
                    
                    # axis formatting
                    axs[i,j].set_xlabel(r'SVM ($N_e = %d$)' % ss, fontsize=11)
                    if j == 0:
                        axs[i,j].set_ylabel(r'%s ($N_e = %d$)' % (method.upper(), ss_oth), 
                                            fontsize=11)
                    axs[i,j].set_xticks(ax_ticks)
                    axs[i,j].set_yticks(ax_ticks)
                    axs[i,j].tick_params(labelsize=10)
                    axs[i,j].set_aspect('equal')
                    axs[i,j].set_xlim(lims)
                    axs[i,j].set_ylim(lims)
                    
            # kom plots
            if method == 'kom':
                # plot parameters
                ax_range = [0.6e-7, 1.4e-4] if kernel == 'linear' \
                           else [-0.002, 0.102]
                ax_ticks = [1e-7, 1e-6, 1e-5, 1e-4] if kernel == 'linear' \
                           else [0.00, 0.04, 0.08]
                           
                # equal ndim soultion
                ndim_oth = res_oth.bal[0]                   
                ind = np.where(res_svm.bal <= ndim_oth)[0][0]
                
                # get corresponding SVM solution
                sdim_sub = sdim_svm.iloc[ind]
                ss = res_svm.iloc[ind].ess
                
                # plot
                axs[i].scatter(np.abs(sdim_sub), np.abs(sdim_oth[1:]), 
                                 s=10, alpha=alpha, zorder=12)
                
                # y = x line
                lims = ax_range
                axs[i].plot(lims, lims, 'k--', lw=1, zorder=0)
                
                # axis formatting
                axs[i].set_xlabel(r'SVM ($N_e = %d$)' % ss, fontsize=10)
                axs[i].set_ylabel(r'%s ($N_e = %d$)' % (method.upper(), ss_oth), 
                                    fontsize=10)
                if kernel == 'linear':
                    axs[i].set_xscale('log')
                    axs[i].set_yscale('log')
                axs[i].tick_params(which='minor', left=False)
                axs[i].set_xticks(ax_ticks)
                axs[i].set_yticks(ax_ticks)
                axs[i].tick_params(labelsize=9)
                axs[i].set_aspect('equal')
                axs[i].set_xlim(lims)
                axs[i].set_ylim(lims)
                    
                    
        if not constrained_layout: fig.tight_layout()
        plt.savefig(join(OUT_DIR,'fig%s.pdf' % (m + 7)), dpi=200, 
                    bbox_inches='tight')
        
