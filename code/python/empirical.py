import numpy as np
import numpy.matlib
import pandas as pd

from argparse import ArgumentParser
from os.path import abspath, dirname, exists, join
from simulations import kom
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from svmpath import SVMPath


# directories
WD = dirname(abspath(__file__))
DATA_DIR = join(WD, *['..', '..', 'data', 'empirical'])
OUT_DIR = join(WD, *['..', '..', 'results'])

# RHC data
RHC_L_PATH = join(DATA_DIR, 'rhc_clean.csv')
RHC_P_PATH = join(DATA_DIR, 'rhc_poly_clean.csv')
# KCB results
KCB_PATH = join(OUT_DIR, 'rhc_kcb_rbf.csv')
# cardmatch results
CARD_L_PATH = join(OUT_DIR, 'rhc_card_linear')
CARD_P_PATH = join(OUT_DIR, 'rhc_card_poly')

# function for computing elbow point
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

# function for gathering statistics about SVM path
def path_stats(path, kernel, eps=1e-6):
    # get useful quantities
    X = path.X
    X_trans = path.X_trans # covariate matrix in transformed space
    y = path.y
    outcomes = path.outcomes
    Q = path.Q
    
    # adjust y to 0-1
    if isinstance(y, np.ndarray):
        y = (y == 1).astype(int)
    else:
        y = (y.to_numpy() == 1).astype(int)

    # outcomes for treated and control
    out1 = outcomes[y == 1].to_numpy()
    out0 = outcomes[y == 0].to_numpy()
        
    # number of treated and control
    n1 = sum(y)
    n0 = len(y) - n1
        
    # prespecified grid of C values
    if kernel == 'linear':
        Cs = np.geomspace(3e-4, 500, 150)
    elif kernel in {'poly', 'polynomial'}:
        Cs = np.geomspace(1.5e-5, 1e3, 150)
    else:
        Cs = np.geomspace(4e-2, 500, 150)
        
    # conditional variance estimate, estimated from 2 neighbors
    if kernel == 'rbf':
        X_sc= StandardScaler().fit_transform(X)
    else:
        X_sc = StandardScaler().fit_transform(X_trans)
    X1 = X_sc[y == 1]
    X0 = X_sc[y == 0]
    nn1 = NearestNeighbors(n_neighbors=2).fit(X1)
    nn0 = NearestNeighbors(n_neighbors=2).fit(X0)

    # conditional variances
    neighb1 = nn1.kneighbors()[1]
    neighb0 = nn0.kneighbors()[1]
    # see Ch. 19 of Imbens & Rubin for conditional variance estimate
    cv1 = ((np.tile(out1, (2,1)).T - out1[neighb1]) ** 2).sum(axis=1) / 4
    cv0 = ((np.tile(out0, (2,1)).T - out0[neighb0]) ** 2).sum(axis=1) / 4

    # save conditional variance
    if not exists(join(OUT_DIR, "cv1.csv")):
        np.savetxt(join(OUT_DIR, "cv1.csv"), cv1 , delimiter=",")
    if not exists(join(OUT_DIR, "cv0.csv")):
        np.savetxt(join(OUT_DIR, "cv0.csv"), cv0 , delimiter=",")           
        
    # containers
    ate = np.zeros(len(Cs))
    se = ate.copy()
    ss = ate.copy()
    ess = ate.copy()
    bal = ate.copy()
    asum = ate.copy()
    
    # pooled variance, can't do dimensional balance stats with rbf
    if kernel != 'rbf':
        sdim = np.zeros((len(Cs), X.shape[-1]))
        bal_sd = ate.copy()
        # pooled variance
        pooled = np.sqrt(X_trans[y == 1].var(axis=0, ddof=1) / 2 + 
                         X_trans[y == 0].var(axis=0, ddof=1) / 2)
    
    # compute statistics for each value of C
    for i, C in enumerate(Cs):
        # sum of weights
        alpha = path.get_alpha(C)
        asum[i] = alpha.sum()
        
        # normalize to sum to 1
        alpha = alpha / (alpha.sum() / 2)
    
        # rescaled weights, sum to number of treated / control units
        w1 = alpha[y == 1] * n1
        w0 = alpha[y == 0] * n0 
        
        # ate estimate
        ate[i] = alpha[y == 1] @ out1 - alpha[y == 0] @ out0
        
        # standard error estimate, see Ch. 19 of Imbens & Rubin
        se[i] = np.sqrt((w1 ** 2 * cv1).sum() / n1 ** 2 + 
                        (w0 ** 2 * cv0).sum() / n0 ** 2)
        
        # subset size
        ss[i] = len(np.where(alpha > eps)[0])
        
        # effective subset size
        ess[i] = alpha[y == 1].sum() ** 2 / (alpha[y == 1] ** 2).sum() + \
                 alpha[y == 0].sum() ** 2 / (alpha[y == 0] ** 2).sum()
        
        # normed difference in means (wrt standardized X)
        bal[i] = np.sqrt(alpha @ Q @ alpha)
        
        # standardized difference in means (not compatible with rbf)
        if kernel != 'rbf': 
            sdim[i] = (alpha * (2*y-1)) @ X_trans / pooled
            # average sdim (per dimension)
            bal_sd[i] = np.abs(sdim[i]).mean()
        
    # build results dataframe
    results = pd.DataFrame({'ate' : ate, 'se' : se, 'bal' : bal, 'ss' : ss, 
                            'ess' : ess, 'asum' : asum},
                           index=Cs)
    if kernel == 'rbf':
        return results
    else:
        # add in average sdim
        results.insert(3, 'bal_sd', bal_sd)
        
        # add in sdim
        temp = pd.DataFrame(sdim, columns=X.columns, index=Cs)
        
        return results.join(temp)
    

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-k','--kernel',choices=['linear','poly','rbf'],
                        default='linear',help='Kernel for specifying function space')
    parser.add_argument('-v','--verbose',action='store_true',default=False,
                        help='Verbosity for output')
    
    return parser.parse_args()

if __name__ == '__main__':
    # command line arguments
    args = parse_arguments()
    kernel = args.kernel 
    verbose = args.verbose
    
    # compute sdim per dimenstion if kernel is not RBF
    compute_sdim = True if kernel != 'rbf' else False

    # read data, split into X and y
    if kernel != 'poly':
        rhc = pd.read_csv(RHC_L_PATH)
        path_kernel = kernel
    else:
        rhc = pd.read_csv(RHC_P_PATH)
        path_kernel = 'linear'
        
    X = rhc.drop(['death','swang1'], axis=1) # unstandardized
    y = rhc.swang1 
    outcomes = rhc.death
    
    # compute path
    print("Computing SVM path... ", end="", flush=True)
    path = SVMPath(X, y, outcomes=outcomes, kernel=path_kernel, verbose=verbose)
    print("Done!\n")
    
    # compute path statistics
    print("Computing path statistics... ", end="", flush=True)
    res = path_stats(path, kernel)
    print("Done!\n")
    
    # compute KOM estimate
    print("Computing kernel optimal weights... ", end="", flush=True)
    res_kom = kom(X, y, outcomes, kernel=path_kernel, compute_sdim=compute_sdim,
                  verbose=verbose) 
    print("Done!\n")
    
    # save results
    res.to_csv(join(OUT_DIR, 'rhc_svm_%s.csv' % kernel))
    res_kom.to_csv(join(OUT_DIR, 'rhc_kom_%s.csv' % kernel))
    
    # compare balance with KOM and cardmatch
    if kernel != 'rbf':    
        # read in cardmatch results
        if kernel == "linear": 
            res_card = pd.read_csv(CARD_L_PATH)
        elif kernel =='poly':
            res_card = pd.read_csv(CARD_P_PATH)
            
        
    # read in kcb results for rbf kernel
    if kernel == 'rbf':
        res_kcb = pd.read_csv(KCB_PATH)
        
    # get SVM point estimates
    bal_max = res.bal.max()
    bal_min = res.bal.min()
    
    # initial solution
    ind1 = 0
    
    if kernel == 'linear': # no elbow point
        ind2 = -1
    else:
        ind2 = elbow(res.asum, res.bal)
        
    # most balanced solution
    ind3 = -1
    
    # comparable to KOM balance
    ind_kom = np.where(res.bal <= res_kom.iloc[0].bal)[0][0]
    
    # comparable to cardmatch balance
    if kernel != 'rbf':
        ind_card = np.where(res.bal <= res_card.iloc[0].bal)[0][0]
    else:
        ind_kcb = np.where(res.bal <= res_kcb.iloc[0].bal)[0][0]
    
    print("Displaying Results for %s kernel...\n" % kernel)
    print("SVM_Unbalanced")
    print("    ATE: %.4f" % res.ate.iloc[ind1])
    print("     SE: %.4f" % res.se.iloc[ind1])
    print("Balance: %.4f" % res.bal.iloc[ind1])
    print("    ESS: %d" % res.ess.iloc[ind1])
    print()
    print("SVM_Elbow")
    print("    ATE: %.4f" % res.ate.iloc[ind2])
    print("     SE: %.4f" % res.se.iloc[ind2])
    print("Balance: %.4f" % res.bal.iloc[ind2])
    print("    ESS: %d" % res.ess.iloc[ind2])
    print()
    print("SVM_Balanced")
    print("    ATE: %.4f" % res.ate.iloc[ind3])
    print("     SE: %.4f" % res.se.iloc[ind3])
    print("Balance: %.4f" % res.bal.iloc[ind3])
    print("    ESS: %d" % res.ess.iloc[ind3])
    print()
    print("SVM_KOM")
    print("    ATE: %.4f" % res.ate.iloc[ind_kom])
    print("     SE: %.4f" % res.se.iloc[ind_kom])
    print("Balance: %.4f" % res.bal.iloc[ind_kom])
    print("    ESS: %d" % res.ess.iloc[ind_kom])
    print()
    if kernel != 'rbf':
        print("CARD")
        print("    ATE: %.4f" % res_card.ate.iloc[0])
        print("     SE: %.4f" % res_card.se.iloc[0])
        print("Balance: %.4f" % res_card.bal.iloc[0])
        print("    ESS: %d" % res_card.ess.iloc[0])
        print()
    else:
        print("KCB")
        print("    ATE: %.4f" % res_kcb.ate.iloc[0])
        print("     SE: %.4f" % res_kcb.se.iloc[0])
        print("Balance: %.4f" % res_kcb.bal.iloc[0])
        print("    ESS: %d" % res_kcb.ess.iloc[0])
        print()
    print("KOM")
    print("    ATE: %.4f" % res_kom.ate.iloc[0])
    print("     SE: %.4f" % res_kom.se.iloc[0])
    print("Balance: %.4f" % res_kom.bal.iloc[0])
    print("    ESS: %d" % res_kom.ess.iloc[0])
    