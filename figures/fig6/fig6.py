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
    
    ## Balance vs. ESS ##

    fig, axs = plt.subplots(1, 3, figsize=(6.5, 4.333), sharey=False, 
                            constrained_layout=constrained_layout)
    if not constrained_layout: fig.subplots_adjust(wspace=0.1)
    
    for i in range(3):
        kernel = kernels[i]
        sub = svm.xs([kernel], level=['kernel'])
        
        # C values
        C = np.unique(sub.index)
        
        # relevant values
        asum = sub.asum
        bal = sub.bal
        ess = sub.ess
        
        # elbow solution
        if kernel == 'linear':
            elb = sub.iloc[-1].ess
        else:
            ind = elbow(asum, bal)
            elb = sub.iloc[ind].ess
           
        # plot curves
        axs[i].plot(ess, bal, label=r'SVM', alpha=1, lw=1.5, zorder=12)
        
        # elbow point
        if kernel != 'linear':
            axs[i].axvline(elb, linestyle='--', color='k', alpha=1, lw=1, zorder=13)
        
        # axis
        if i == 1: axs[i].set_xlabel(r'Effective sample size', fontsize=12)
            
        if i == 0:
            axs[i].set_xticks([3500, 3750, 4000, 4250])
            axs[i].set_ylabel(r'Normed difference-in-means', fontsize=12)
        axs[i].tick_params(labelsize=10)
        axs[i].set_aspect(1 / axs[i].get_data_ratio())    
        axs[i].grid(True)
        
    # layout and save
    fig.align_ylabels()
    if not constrained_layout: fig.tight_layout()
    plt.savefig(join(OUT_DIR,'fig6.pdf'), dpi=200, bbox_inches='tight')