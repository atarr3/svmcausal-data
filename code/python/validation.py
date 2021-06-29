import gurobipy as gp
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from gurobipy import GRB
from numpy.random import rand, randn
from os.path import abspath, dirname, join
from svmpath import SVMPath

# generate continuous random variable correlated to variable x by rho
def sample_cor(x, rho):
    y = (rho * (x - x.mean())) / x.std() + np.sqrt(1 - rho ** 2) * randn(len(x))
        
    return y

# generate simulation datasets
def gen_data(size, scenario):
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
    y = a0 + a1*w1 + a2*w2 + a3*w3 + a4*w4 +a5*w8 + a6*w9 + a7*w10 + g1*z 
    
    # dataframe
    df = pd.DataFrame({'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 
                       'w6': w6, 'w7': w7, 'w8': w8, 'w9': w9, 'w10': w10,
                       'Z' : z, 'Y': y})
    return df

# callback function
def mycallback(model, where):
    # current time limit
    tlim = model.getParamInfo('TimeLimit')[2]
    # check solution status
    if where == GRB.Callback.MIP:
        time = model.cbGet(GRB.Callback.RUNTIME)
        count = model.cbGet(GRB.Callback.MIP_SOLCNT)
        # terminate if past time limit with solution found
        if time > tlim and count > 0:
            model.terminate()

# compute integer solution over SVM path breakpoints
def integer_sol(path, tlim=30, verbose=False, save=False):
    # path quantities
    Q = path.Q
    y = path.y
    Es = path.E
    Ls = path.L
    Cs = path.C
    alphas = path.alpha
    
    # arrays
    obj_svm = np.zeros(len(Es))
    obj_qip = obj_svm.copy()
    ss_svm = obj_svm.copy()
    ss_qip = obj_svm.copy()
    bal_svm = obj_svm.copy()
    bal_qip = obj_svm.copy()
    cov = obj_svm.copy()
    gap = obj_svm.copy()
    prev_sol = []
    
    # iterate through path breakpoints
    for i, (alpha, C, E, L) in enumerate(zip(alphas, Cs, Es, Ls)):
        print('\tSolving problem %d of %d... ' % (i+1,len(Cs)), end='', flush=True)
        
        m = gp.Model("qip")
        a = m.addMVar(len(Q), vtype='B', name="alpha")
        
        # initialize a with previous solution
        if len(prev_sol) > 0: a.start = prev_sol
        
        # use svm solution as hint
        a.varhintval = np.rint(alpha)
        
        # # assign priority (0 for marginal SV, 1 otherwise)
        # a.varhintpri = 1 - E
    
        #  set constraint
        m.addConstr(a @ y == 0)
        # set objective
        m.setObjective(a @ (Q * C / 2) @ a - a.sum(), GRB.MINIMIZE)
    
        # optimize
        m.setParam("OutputFlag",0)
        m.setParam("TimeLimit", tlim)
        m.optimize()
        # store MIP gap (upper - lower) / |upper|
        gap[i] = m.MIPGap
        print('solution found with gap %.4f%%' % (gap[i] * 100), end='\n')
        # round to nearest integer
        alpha_qip = np.rint(np.abs(a.x))
        # update prev_sol
        prev_sol = alpha_qip
        
        # support vector sets
        sv_qip = np.where(alpha_qip > 0)[0]
        sv = np.where(E | L)[0]
        
        # coverage
        if len(sv_qip) > 0:
            cov[i] = len(set(sv_qip).intersection(set(sv))) / len(sv_qip)
        else:
            cov[i] = 0
        
        # alpha sums
        ss_svm[i] = alpha.sum()
        ss_qip[i] = alpha_qip.sum()
        
        # quadratic form
        qf_svm = alpha @ Q @ alpha
        qf_qip = alpha_qip @ Q @ alpha_qip
        
        # norm DIM
        bal_svm[i] = np.sqrt(qf_svm) / (ss_svm[i] / 2)
        bal_qip[i] = np.sqrt(qf_qip) / (ss_qip[i] / 2) if ss_qip[i] > 0 else -1
        
        # objective value
        obj_svm[i] = (C / 2) * qf_svm - ss_svm[i]
        obj_qip[i] = (C / 2) * qf_qip - ss_qip[i]

    # store results in data frame
    res = pd.DataFrame({'C' : Cs, 'obj_svm' : obj_svm, 'obj_qip' : obj_qip,
                        'bal_svm' : bal_svm, 'bal_qip' : bal_qip, 
                        'coverage' : cov, 'gap' : gap})
    
    return res        

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-k','--kernel', choices=['linear','poly','polynomial','rbf'],
                        default='linear', help='Kernel for specifying function space')
    parser.add_argument('-t', '--time-limit', type=int, default=300, 
                        help='Time limit for computing each integer solution. '
                        'The default value is 300 seconds')
    
    return parser.parse_args()

if __name__ == '__main__':
    # command line arguments
    args = parse_arguments()
    kernel, tlim = args.kernel, args.time_limit
    
    # directory of file
    WD = dirname(abspath(__file__))
    
    # output directory
    OUT_DIR = join(WD, *['..', '..', 'results'])
    
    # generate data
    np.random.seed(3000)
    data = gen_data(500, "G")
    
    # split into covariates, treatment, and outcome
    X = data.drop(['Z','Y'], axis=1) # unstandardized
    y = (2 * data.Z - 1).to_numpy()
    outcomes = data.Y.to_numpy()
    
    # compute path
    print('Computing path...')
    path = SVMPath(X, y, outcomes, kernel=kernel)
    print('Computing path... Done!')
    
    # compute integer solutions
    print('Computing integer solutions...')
    results = integer_sol(path, tlim=tlim, save=True)
    print('Computing integer solutions... Done!')
    
    # save results
    results.to_csv(join(OUT_DIR, 'validation_%s.csv' % kernel), index=False)
    