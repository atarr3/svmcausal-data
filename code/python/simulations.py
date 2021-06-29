import gurobipy as gp
import numpy as np
import numpy.matlib
import pandas as pd

from argparse import ArgumentParser
from gurobipy import GRB
from numpy.random import rand, randn
from os.path import abspath, dirname, join
from sklearn.cluster import estimate_bandwidth
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, RBF, WhiteKernel
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from svmpath import SVMPath

# directories
WD = dirname(abspath(__file__))
DATA_DIR = join(WD, *['..', '..', 'data', 'simulations'])
OUT_DIR = join(WD, *['..', '..', 'results'])

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

# function for computing weights via kernel optimal matching (Kallus 2018)
def kom(X, y, outcomes, kernel='linear', compute_sdim=False, eps=1e-6,
        verbose=False):    
    # check that y is binary
    if len(np.unique(y)) != 2: 
        raise ValueError("class label vector `y` is not binary")
        
    # data info and useful vectors
    n = len(y)
    en = np.ones(n) / n
    
    # transform y to {-1,1} if not already
    if y.min() != -1: y = 2*y - 1
    
    # number of treated and contrl
    n1 = sum(y == 1)
    n0 = n - n1
    
    # outcomes for treated and control
    out1 = outcomes[y == 1].to_numpy()
    out0 = outcomes[y == -1].to_numpy()
    
    # copy of covariate matrix
    X_c = X.copy()
    
    # set up kernels
    if kernel not in {'linear', 'poly', 'polynomial', 'rbf'}: 
        raise ValueError("invalid kernel")
        
    if kernel == 'linear':
        kern = ConstantKernel() * (DotProduct(sigma_0=0, 
                                              sigma_0_bounds='fixed')) + \
               WhiteKernel()
    elif kernel == 'rbf':
        # compute gamma via median heuristic
        gamma = estimate_bandwidth(StandardScaler().fit_transform(X), quantile=0.5)
        kern = ConstantKernel() * RBF(length_scale=gamma/np.sqrt(2), 
                                      length_scale_bounds='fixed') + \
               WhiteKernel()
    else: # degree 2 polynomial
        # identify binary-valued columns
        if isinstance(X, np.ndarray):
            is_bin = np.where([len(np.unique(col)) == 2 for col in X.T])[0]
        else:
            is_bin = np.where([len(X[col].unique()) == 2 for col in X])[0]
            
        # number of features and number of pairs
        d = X.shape[1]
        # index of square term for feature i in PolynomialFeatures
        sq_ind = np.append([d], d + np.cumsum([d-i for i in range(d-1)]))
    
        X = PolynomialFeatures(2, include_bias=False).fit_transform(X)
        # drop squared terms for binary features
        X = np.delete(X, sq_ind[is_bin], axis=1)
        
        # drop redundant features (caused by categorical variables)
        X = X[:, X.std(0) > 0]
        
        # linear kernel with polynomial features
        kern = ConstantKernel() * (DotProduct(sigma_0=0, 
                                              sigma_0_bounds='fixed')) + \
               WhiteKernel()
                 
    # standardize covariate matrix
    X = StandardScaler().fit_transform(X)
    
    # compute gram matrix
    if kernel != 'rbf':
        K = pairwise_kernels(X)
    else:
        K = pairwise_kernels(X, metric='rbf', gamma=gamma**-2)
    
    # initialize variance matrix diagonal
    diag = np.ones(len(y))   
    
    # estimate noise variance from GPML for control outcomes
    gpr0 = GaussianProcessRegressor(kernel=kern, normalize_y=True
                                   ).fit(X[y == -1], outcomes[y == -1])
    diag[y == -1] = gpr0.kernel_.k2.noise_level 
    # estimate noise variance from GPML for treated outcomes
    gpr1 = GaussianProcessRegressor(kernel=kern, normalize_y=True
                                   ).fit(X[y == 1], outcomes[y == 1])
    diag[y == 1] = gpr1.kernel_.k2.noise_level
    
    # variance matrix
    W = np.diag(diag)
    
    # set up optimization problem
    m = gp.Model("init")
    a = m.addMVar(n, lb=0, ub=1, vtype='C', name="alpha")
    
    #  set constraints
    m.addConstr(a[y == 1].sum() == 1)
    m.addConstr(a[y == -1].sum() == 1)
    
    # indicator matrices
    I1 = np.diag(y==1).astype(int)
    I0 = np.diag(y==-1).astype(int)
    
    # kernel submatrices
    K1 = I1 @ K @ I1
    K0 = I0 @ K @ I0
    
    # set objective
    m.setObjective(a @ (K1 + K0 + W) @ a - 2 * en @ K @ a, GRB.MINIMIZE)
    
    # solve and get solution
    if not verbose: m.setParam("OutputFlag", 0)
    m.setParam('BarConvTol', 1e-11)
    m.setParam('PSDTol', 2e-5) # shouldn't be having psd issues
    m.optimize()
    
    # compute weights
    alpha = np.array(a.x)
    
    # compute ATE via weighted DIM
    treat = (y == 1) * alpha
    control = (y == -1) * alpha  
    ate = treat @ outcomes - control @ outcomes
    
    # conditional variance estimate, estimated from 2 neighbors
    X1 = X[y == 1]
    X0 = X[y == -1]
    nn1 = NearestNeighbors(n_neighbors=2).fit(X1)
    nn0 = NearestNeighbors(n_neighbors=2).fit(X0)

    # conditional variances
    neighb1 = nn1.kneighbors()[1]
    neighb0 = nn0.kneighbors()[1]
    # see Ch. 19 of Imbens & Rubin for conditional variance estimate
    cv1 = ((np.tile(out1, (2,1)).T - out1[neighb1]) ** 2).sum(axis=1) / 4
    cv0 = ((np.tile(out0, (2,1)).T - out0[neighb0]) ** 2).sum(axis=1) / 4
    
    # standard error estimate
    w1 = alpha[y == 1] * n1
    w0 = alpha[y == -1] * n0
    se = np.sqrt((w1 ** 2 * cv1).sum() / n1 ** 2 + 
                 (w0 ** 2 * cv0).sum() / n0 ** 2)
    
    # subset size
    ss = len(np.where(alpha > eps)[0])
    
    # effective sample size
    ess = alpha[y == 1].sum() ** 2 / (alpha[y == 1] ** 2).sum() + \
          alpha[y == -1].sum() ** 2 / (alpha[y == -1] ** 2).sum()
    
    # normed difference in means
    bal = np.sqrt(alpha @ (K * np.outer(y, y)) @ alpha)
    
    results = pd.DataFrame({'ate' : ate, 'se' : se, 'bal' : bal, 'ss' : ss, 
                            'ess' : ess,},
                           index = [0])
    
    if kernel == 'rbf' or not compute_sdim:
        return results
    elif compute_sdim:
        # compute sdim
        pooled = np.sqrt(X_c[y == 1].var(axis=0, ddof=1) / 2 + 
                         X_c[y == -1].var(axis=0, ddof=1) / 2)
        sdim = (alpha * y) @ X_c / pooled
        
        # average sdim
        bal_sd = np.abs(sdim).mean()
        results.insert(3, 'bal_sd', bal_sd)
        
        # temp data frame
        temp = pd.DataFrame([sdim], columns=X_c.columns)
        
        return results.join(temp)

# Kallus simulations
def gen_data_kallus(size, beta=0.1, model='linear', misspec=False):
    # covariates
    X = randn(size, 2)
    
    # misspecification scenario (X1, X2 transformation)
    if misspec:
        Z1 = (2 + X[:,0]) / np.exp(X[:, 0])
        Z2 = (X[:,0] * X[:,1] / 25) ** 3
        X[:,0] = Z1
        X[:,1] = Z2
        
    # propensity scores
    if model == 'linear':
        prop = 1 / (1 + np.exp(-beta * X.sum(axis=1)))
    else:
        prop = 1 / (1 + np.exp(-beta * (X.sum(axis=1) + (X ** 2).sum(axis=1)
                                        + X[:,0]*X[:,1])))
    
    # treatment assignments
    prob = rand(size)
    T = (prop > prob).astype(int)
    
    # outcomes (true effect is 1)
    if model == 'linear':
        Y = T + X.sum(axis=1) + randn(size)
        # demean
        Y = Y - Y.mean()
    else:
        Y = T + X.sum(axis=1) + (X ** 2).sum(axis=1) + X[:,0]*X[:,1] + randn(size)
        # demean
        Y = Y - Y.mean() 
        
    # dataframe
    df = pd.DataFrame({'x1' : X[:,0], 'x2' : X[:,1], 'Z' : T, 'Y': Y})
    
    return df
    
    
# Chan & Wong simulations
def gen_data_wong(size, scenario=None):
    # covariates
    Z = randn(size, 10)
    # transformed covariates (observed)
    X = np.column_stack((
                np.exp(Z[:,0])/2, 
                Z[:,1] / (1 + np.exp(Z[:,0])),
                (Z[:,0] * Z[:,2] / 25 + 0.6) ** 3,
                (Z[:,1] + Z[:,3] + 20) ** 2,
                Z[:,4], Z[:,5], Z[:,6], Z[:,7], Z[:,8], Z[:,9] 
                       ))
    # propensity score
    prop = np.exp(-Z[:,0] - 0.1*Z[:,3]) / (1 + np.exp(-Z[:,0] - 0.1*Z[:,3]))
    
    # treatment assignments
    prob = rand(size)
    T = (prop > prob).astype(int)
    
    # outcomes (true effect is 10)
    Y0 = 200 - 0.5 * (27.4*Z[:,0] + 13.7*Z[:,1] + 13.7*Z[:,2] + 
                      13.7*Z[:,3]) + randn(size)
    Y1 = Y0 + 10 + 1.5 * (27.4*Z[:,0] + 13.7*Z[:,1] + 13.7*Z[:,2] + 
                              13.7*Z[:,3])
    
    Y = np.where(T == 1, Y1, Y0)
    
    df = pd.DataFrame(X, columns=['x' + str(num) for num in range(1,11)])
    df['Z'] = T
    df['Y'] = Y
    
    return df

# generate continuous random variable correlated to variable x by rho
def sample_cor(x, rho):
    y = (rho * (x - x.mean())) / x.std() + np.sqrt(1 - rho ** 2) * randn(len(x))
        
    return y

# Setoguchi simulations (Lee et al.)
def gen_data_lee(size, scenario):
    # data generation coefficients
    b0, b1, b2, b3, b4, b5, b6, b7 = 0, 0.8, -0.25, 0.6, -0.4, -0.8, -0.5, 0.7
    a0, a1, a2, a3, a4, a5, a6, a7 = -3.85, 0.3, -0.36, -0.73, -0.2, 0.71, -0.19, 0.26
    g1 = -0.4
        
    # covariates
    w1 = randn(size)
    w2 = randn(size)
    w3 = randn(size)
    w4 = randn(size)
    w5 = sample_cor(w1, 0.2)
    w6 = sample_cor(w2, 0.9)
    w7 = randn(size)
    w8 = sample_cor(w3, 0.2)
    w9 = sample_cor(w4, 0.9)
    w10 = randn(size)
    
    # dichotomize
    w1 = (w1 > w1.mean()).astype(float)
    w3 = (w3 > w3.mean()).astype(float)
    w5 = (w5 > w5.mean()).astype(float)
    w6 = (w6 > w6.mean()).astype(float)
    w8 = (w8 > w8.mean()).astype(float)
    w9 = (w9 > w9.mean()).astype(float)
    
    # propensity scores
    if scenario == "A":
        z_trueps = (1 + np.exp( -(b0 + b1*w1 + b2*w2 + b3*w3 + b4*w4 + b5*w5 + 
                                  b6*w6 + b7*w7) ) ) ** -1
    elif scenario == "B":
        z_trueps = (1 + np.exp( -(b0 + b1*w1 + b2*w2 + b3*w3 + b4*w4 + b5*w5 + 
                                  b6*w6 + b7*w7 + b2*w2*w2) ) ) ** -1
    elif scenario == "C":
        z_trueps = (1 + np.exp( -(b0 + b1*w1 + b2*w2 + b3*w3 + b4*w4 + b5*w5 + 
                                  b6*w6 + b7*w7 + b2*w2*w2 +b4*w4*w4 + b7*w7*w7) ) ) ** -1
    elif scenario == "D":
        z_trueps = (1 + np.exp( -(b0 + b1*w1 + b2*w2 + b3*w3 + b4*w4 + b5*w5 + 
                                  b6*w6 + b7*w7 + b1*0.5*w1*w3 + b2*0.7*w2*w4 
                                  + b4*0.5*w4*w5 + b5*0.5*w5*w6) ) )^-1
    elif scenario == "E":
        z_trueps = (1 + np.exp( -(b0 + b1*w1 + b2*w2 + b3*w3 + b4*w4 + b5*w5 + 
                                  b6*w6 + b7*w7 + b2*w2*w2 + b1*0.5*w1*w3 + 
                                  b2*0.7*w2*w4 + b4*0.5*w4*w5 + b5*0.5*w5*w6) ) ) ** -1
    elif scenario == "F":
        z_trueps = (1 + np.exp( -(b0 + b1*w1 + b2*w2 + b3*w3 + b4*w4 + b5*w5 + 
                                  b6*w6 + b7*w7 + b1*0.5*w1*w3 + b2*0.7*w2*w4 
                                  + b3*0.5*w3*w5 + b4*0.7*w4*w6 + b5*0.5*w5*w7
                                  + b1*0.5*w1*w6 + b2*0.7*w2*w3 + b3*0.5*w3*w4 
                                  + b4*0.5*w4*w5 + b5*0.5*w5*w6) ) ) ** -1
    else:
     # scenario G
        z_trueps = (1 + np.exp( -(b0 + b1*w1 + b2*w2 + b3*w3 + b4*w4 + b5*w5 + 
                                  b6*w6 + b7*w7 + b2*w2*w2 + b4*w4*w4 + 
                                  b7*w7*w7 + b1*0.5*w1*w3 + b2*0.7*w2*w4 + 
                                  b3*0.5*w3*w5 + b4*0.7*w4*w6 + b5*0.5*w5*w7 + 
                                  b1*0.5*w1*w6 + b2*0.7*w2*w3 + b3*0.5*w3*w4 +
                                  b4*0.5*w4*w5 + b5*0.5*w5*w6) ) ) ** -1
        
    # random draw
    prob_exposure = rand(size)

    # treatment assignment
    z = (z_trueps > prob_exposure).astype(int)
    
    # outcome
    y = a0 + a1*w1 + a2*w2 + a3*w3 + a4*w4 +a5*w8 + a6*w9 + a7*w10 + g1*z + \
        np.sqrt(0.1) * randn(size)
    
    # dataframe
    df = pd.DataFrame({'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 
                       'w6': w6, 'w7': w7, 'w8': w8, 'w9': w9, 'w10': w10,
                       'Z' : z, 'Y': y})
    return df

# function for conducting PLoS One simulations / Chan & Wong simulations
def simulations(scenario, kernel='linear', nsamples=500, ntrials=1000, 
                seed=2002, verbose=False, save=False,
                save_dir=OUT_DIR):
    # set seed for repeatability
    np.random.seed(seed)
    
    # keeps track of which trials failed
    fail_ind = []
    
    # grid of C values to compute ATE estimates at
    if kernel == 'linear': 
        Cs = np.geomspace(1e-3, 10, 100)
    elif kernel == 'rbf': 
        Cs = np.geomspace(1e-1, 1e3, 100)
    else: # polynomial
        if scenario == "cw":
            Cs = np.geomspace(1e-3, 100, 100)
        else:
            Cs = np.geomspace(1e-3, 10, 100)
        
    # simulation setup
    if scenario != "cw":
        gen_data = gen_data_lee
    else:
        gen_data = gen_data_wong
    
    # initialize ate matrix
    ate = np.zeros((ntrials, len(Cs)))
    # initialize standard error estimate matrix
    se = ate.copy()
    # initialize balance matrix
    balance = ate.copy()
    # initialize weight sum matrix
    asum = ate.copy()
    # initialize subset size matrix
    ss = ate.copy()
    # initialize effective subset size matrix
    ess = ate.copy()
    # initialize KOM vectors
    ate_kom = np.zeros(ntrials)
    se_kom = ate_kom.copy()
    bal_kom = ate_kom.copy()
    ss_kom = ate_kom.copy()
    ess_kom = ate_kom.copy()
    
    # pandas multi-index
    index = pd.MultiIndex.from_product([np.arange(ntrials)+1, Cs], 
                                       names=['trial','C'])
    
    # conduct simulation
    for t in range(ntrials):
        print('Trial %d of %d' % (t+1, ntrials), end='\r', flush=True)
        # generate data
        data = gen_data(nsamples, scenario)
        if save: data.to_csv(join(DATA_DIR,'data_%s_%s.csv' % (scenario,t+1)),
                             index=False)
        
        # split into covariates, treatment, and outcome
        X = data.drop(['Z','Y'], axis=1) # unstandardized
        y = (2 * data.Z - 1).to_numpy()
        outcome = data.Y
        
        n1 = (y == 1).sum()
        n0 = (y == -1).sum()
        
        # compute path
        try:
            path = SVMPath(X, y, outcome, kernel=kernel, verbose=verbose)
        except:
            fail_ind.append(t)
            continue
        
        # get weights at C values and compute ATE
        for i, C in enumerate(Cs):
            try:
                alpha = path.get_alpha(C)
            except ValueError:
                # can't obtain weights, set to nan
                ate[t, i] = np.nan
                se[t, i] = np.nan
                ss[t, i] = np.nan
                ess[t, i] = np.nan
                balance[t, i] = np.nan
                asum[t, i] = np.nan
                continue
            
            # compute ATE
            treat = (y == 1) * alpha
            control = (y == -1) * alpha
            nt = treat.sum() # weight sum
            nc = control.sum() # weight sum       
            ate[t, i] = treat @ outcome / nt - control @ outcome / nc
            
            # conditional variance estimate, estimated from 2 neighbors
            if kernel == 'rbf':
                X_trans = StandardScaler().fit_transform(path.X)
            else:
                X_trans = StandardScaler().fit_transform(path.X_trans)
            X1 = X_trans[y == 1]
            X0 = X_trans[y == -1]
            nn1 = NearestNeighbors(n_neighbors=2).fit(X1)
            nn0 = NearestNeighbors(n_neighbors=2).fit(X0)

            # conditional variances
            out1 = outcome[y == 1].to_numpy()
            out0 = outcome[y == -1].to_numpy()
            neighb1 = nn1.kneighbors()[1]
            neighb0 = nn0.kneighbors()[1]
            # see Ch. 19 of Imbens & Rubin for conditional variance estimate
            cv1 = ((np.tile(out1,(2,1)).T - out1[neighb1]) ** 2).sum(axis=1) / 4
            cv0 = ((np.tile(out0,(2,1)).T - out0[neighb0]) ** 2).sum(axis=1) / 4
            
            # standard error estimate, see Ch. 19 of Imbens & Rubin
            w1 = alpha[y == 1] / nt * n1
            w0 = alpha[y == -1] / nc * n0 
            se[t, i] = np.sqrt((w1 ** 2 * cv1).sum() / n1 ** 2 + 
                                (w0 ** 2 * cv0).sum() / n0 ** 2)                
            
            # compute balance
            balance[t, i] = np.sqrt(alpha @ path.Q @ alpha) / (alpha.sum() / 2)
            
            # compute weight sum
            asum[t, i] = alpha.sum()
            
            # compute subset size
            ss[t, i] = len(alpha[alpha > 1e-6])
            
            # compute effective subset size
            ess[t, i] = alpha[y == 1].sum() ** 2 / (alpha[y == 1] ** 2).sum() + \
                        alpha[y == -1].sum() ** 2 / (alpha[y == -1] ** 2).sum()
                        
        # compute ate via KOM
        res = kom(X, y, outcome, kernel=kernel, compute_sdim=False)
        ate_kom[t], se_kom[t], bal_kom[t] = res.ate, res.se, res.bal 
        ss_kom[t], ess_kom[t] = res.ss, res.ess
        
    # advance to next line
    print()            
    print('Failed on %d trials' % len(fail_ind))
    
    # build results dataframe
    results = pd.DataFrame({'ate' : ate.flatten(), 'se' : se.flatten(),
                            'bal' : balance.flatten(), 
                            'ss' : ss.flatten(), 'ess' : ess.flatten(),
                            'asum' : asum.flatten()}, index=index)
    
    results_kom = pd.DataFrame({'ate' : ate_kom, 'se' : se_kom,'bal' : bal_kom,
                                'ss' : ss_kom, 'ess' : ess_kom}, 
                               index=pd.Index(np.arange(1000)+1, name='trial'))
    
    return results, results_kom

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-s','--scenario',choices=['A','E','G',"cw"], default='G',
                        help="Which scenario of the simulations to use. The "
                        "default is 'G'")
    parser.add_argument('-k','--kernel',choices=['linear','poly','rbf'],
                        default='linear',help='Kernel for specifying function space')
    parser.add_argument('-sd','--save-data',action='store_true',default=False,
                        help='Save data to default location in simulations()')
    parser.add_argument('-v','--verbose',action='store_true',default=False,
                        help='Verbosity for output')
    
    return parser.parse_args()

if __name__ == '__main__':
    # command line arguments
    args = parse_arguments()
    scenario, kernel = args.scenario, args.kernel
    verbose = args.verbose
    save = args.save_data
    
    # setoguchi simulations
    if scenario == "G":
        # set seed
        seed = 4002
            
        # true effect
        tau = -0.4
        
        # sample size
        nsamples = 500
        
    # chan & wong simulations
    elif scenario == "cw":
        # set seed
        seed = 5000
        
        # true effect
        tau = 10
        
        # sample size
        nsamples = 500
        
        
    # compute ate
    res, res_kom = simulations(scenario, nsamples=nsamples, kernel=kernel,
                               seed=seed, verbose=verbose, save=save)
        
    # save results
    res.to_csv(join(OUT_DIR, '%s_svm_%s.csv' % (scenario, kernel)))
    res_kom.to_csv(join(OUT_DIR, '%s_kom_%s.csv' % (scenario, kernel)))
    