import gurobipy as gp
import numpy as np

from gurobipy import GRB
from scipy.linalg import cho_solve, solve_triangular
from scipy.linalg.blas import drotg
from sklearn.cluster import estimate_bandwidth
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from timeit import default_timer
from warnings import warn

class SVMPath(object):
    def __init__(self, X, y, outcomes, lambda_min=1e-3, kernel='linear', 
                 eps=1e-6, verbose=False):
        # set data
        self.X = X.copy()
        self.y = y.copy()
        self.outcomes = outcomes
        self.kernel = kernel
        self.lambda_min = lambda_min
        self.separated = False # flag for separable data
        
        # check that y is binary
        if len(np.unique(y)) != 2: 
            raise ValueError("class label vector `y` is not binary")
        
        # transform y to {-1,1} if not already
        if y.min() != -1: y = 2*y - 1
        
        # convert to numpy if pandas type
        if not isinstance(y, np.ndarray): y = y.to_numpy()
        
        # data attributes
        n = len(y)
        n_p = (y.sum() + n) // 2
        n_n = n - n_p
        
        # compute signed kernel matrix [y_i y_j K(x_i, x_j)]
        if kernel not in {'linear', 'poly', 'polynomial', 'rbf'}: 
            raise ValueError("invalid kernel")
            
        # form polynomial feature matrix if kernel is polynomial
        if kernel in {'poly', 'polynomial'}:
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
            
        # transformed feature matrix (unstandardized)
        self.X_trans = X if kernel != 'rbf' else None
            
        # standardize covariate matrix
        X = StandardScaler().fit_transform(X)
        
        # compute signed kernel matrix
        if kernel != 'rbf':
            Q = pairwise_kernels(X) * np.outer(y, y)
        else:
            # compute gamma via median heuristic
            gamma = estimate_bandwidth(X, quantile=0.5)
            Q = pairwise_kernels(X, metric='rbf', gamma=gamma**-2) * \
                np.outer(y, y)
            
        self.Q = Q
        
        # initialize path
        s = default_timer()
        
        # positive majority
        if n_p >= n_n:
            lambda_t, mu_t, alpha_t, E, L, R = init_path(Q,y,eps=eps,
                                                         verbose=verbose)
        # negative majority
        else:
            # make positive class majority by inverting y
            lambda_t, mu_t, alpha_t, E, L, R = init_path(Q,-y,eps=eps,
                                                         verbose=verbose)
            # correct mu0 to account for y sign change
            mu_t = -mu_t
            
        t_init = default_timer() - s
            
        # initialize arrays and values
        niter = 0
        events = []
        size = 4*n
        C = np.full(size, 1/lambda_t)
        alpha, beta0 = np.full((size, n),alpha_t), np.full(size, mu_t/lambda_t)
        dalphas = np.zeros((size, n))
        dmus = C.copy()
        Es = np.full((size, n), E)
        Ls = np.full((size, n), L)
        Rs = np.full((size, n), R)
        obj_dink = C.copy()
        obj_svm = C.copy()
        
        # initial objective values
        t1 = alpha_t @ Q @ alpha_t
        t2 = alpha_t.sum()
        obj_svm[0] = t1 / (2*lambda_t) - t2
        obj_dink[0] = np.sqrt(t1) - lambda_t / 2 * t2
        
        # initial function outputs: y_i f(x_i) = y_i (beta'x_i + beta0)
        yf = (Q.dot(alpha_t) + y * mu_t) / lambda_t
        
		# array of sample indices in the order that they were added along the
		# regularization path (i.e., in order of update_cholesky calls)
        E_ind = np.where(E)[0]
		
        # initialize Cholesky decomposition
        U = np.array([])
        for i in range(len(E_ind)):
            U = update_cholesky(U, Q[np.ix_(E_ind[:i],E_ind[:i])].squeeze(),
                                y[E_ind[:i]], Q[E_ind[:i+1], E_ind[i]], y[E_ind[i]])
       
        # compute the regularization path
        s = default_timer()
        while lambda_t > lambda_min and not self.separated:
            # update iteration count
            niter += 1
            if niter % 100 == 0 and verbose: 
                print("iter %d: lambda = %.3e\r" % (niter, lambda_t))

            # compute search directions
            dalpha, dmu = compute_dirs(U, Q[np.ix_(E_ind,E_ind)], y[E_ind])
			# reorder dalpha in order of samples
            dalpha = np.atleast_1d(dalpha)[E_ind.argsort()]
            # change in y_i f(x_i)
            yh = Q[E].T.dot(dalpha) + y * dmu
            
            # updata derivative arrays
            dmus[niter-1] = dmu # derivative between C[i] and C[i+1]
            dalphas[niter-1][E] = dalpha
            
            # compute next breakpoint
            
            # E -> (L or R)
            delta_E = np.full(len(E_ind), -np.inf)
            delta_E[dalpha > 0] = -alpha_t[E][dalpha > 0] / dalpha[dalpha > 0]
            delta_E[dalpha < 0] = (1 - alpha_t[E][dalpha < 0]) / dalpha[dalpha < 0]
            
            # (L or R) -> E   
            #
            # find L/R indices which don't remain in L/R for any lambda. There
            # are three conditions to check (see (Sentelle 2016) for details):
            #
            # 1. y - h(x) = 0 => y f(x) > 1 (x in R) or y f(x) < 1 (x in L)
            # 
            # 2. y f(x) is decreasing in lambda decreasing => y f(x) < 1 (x in L)
            #
            # 3. y f(x) is increasing in lambda decreasing => y f(x) > 1 (x in R)
            #
            elig = ~((np.abs(1 - yh) < eps) | 
                     ((yf - yh < -eps) & L) | 
                     ((yf - yh > eps) & R) | E)
          
            # compute deltas for y_i f(x_i) -> 1
            deltas = np.divide(lambda_t * (yf - 1), 1 - yh, 
                               out=np.full(n,-np.inf), where=elig)
            # retain all deltas < eps. note that this includes small, positive
            # delta corresponding to repeat events with some numerical err
            deltas[deltas / lambda_t >= eps] = -np.inf
            # update with delta_E values
            deltas[E] = delta_E
            
            # # check for ties by comparing E -> (L or R) events with 
            # # (L or R) -> E events, choosing the event corresponding to the
            # # smallest index
            # E_max_ind = E_ind[deltas[E].argmax()]
            # delta_E_max = deltas[E_max_ind]
            # LR_max_ind = np.where(~E)[0][deltas[~E].argmax()]
            # delta_LR_max = deltas[LR_max_ind]
            
            # if (delta_E_max > delta_LR_max) or (E_max_ind < LR_max_ind) and
            #     (np.abs(delta_E_max - delta_LR_max) / delta_LR_max < 1e-9): 
            #     # index relative to whole sample
            #     change_ind = E_max_ind
            #     # index relative to margin set
            #     change_ind_E = np.where(E_ind == change_ind)[0].squeeze()
            #     # get maximum delta (smallest negative)
            #     delta_max = delta_E_max
            # else:
            #     change_ind = LR_max_ind
            #     delta_max = delta_LR_max
                   
            # index corresponding to smallest change (relative to whole sample)
            change_ind = deltas.argmax()
            # index relative to margin set
            change_ind_E = np.where(E_ind == change_ind)[0].squeeze()
            # get maximum delta (smallest negative)
            delta_max = deltas[change_ind]
                   
            # update values
            yf = lambda_t / (lambda_t + delta_max) * (yf - yh) + yh
            lambda_t += delta_max
            alpha_t[E] += delta_max * dalpha
            mu_t += delta_max * dmu
            
            # check if new lambda is below lambda_min
            if lambda_t < lambda_min:
                self.lambda_final = lambda_t
                niter -= 1
                if verbose: 
                    print("Minimum lambda reached... terminating path\n")
                break
            
            # update sets            
            if E[change_ind]:
				#  E -> R event (alpha_t -> 0)
                if alpha_t[change_ind] < eps:
                    events.append('%4d: E -> R' % change_ind)
                    E[change_ind] = False
                    R[change_ind] = True
				#  E -> L event (alpha_t -> 1)
                else:
                    events.append('%4d: E -> L' % change_ind)
                    E[change_ind] = False
                    L[change_ind] = True
                    
                # downdate Cholesky decomposition
                U = downdate_cholesky(U, y[E_ind], change_ind_E)
                # update E_ind by removing value at change_ind_E
                E_ind = np.delete(E_ind, change_ind_E)
            else:
                # R -> E event
                if R[change_ind]:
                    events.append('%4d: R -> E' % change_ind)
                    E[change_ind] = True
                    R[change_ind] = False
                # L -> E event
                else:
                    events.append('%4d: L -> E' % change_ind)
                    E[change_ind] = True
                    L[change_ind] = False
                    
                # update Cholesky decomposition
                U = update_cholesky(U, Q[np.ix_(E_ind,E_ind)], y[E_ind], 
                                    Q[np.append(E_ind, change_ind),change_ind],
                                    y[change_ind])
                # update E_ind by appending change_ind to the end
                E_ind = np.append(E_ind, change_ind)
                
            # check if data separated by solution
            if not any(L):
                self.separated = True
                if verbose:
                    print("Data separated with kernel `%s`... terminating path\n" %
                          kernel)
                
            # remove negative weights from alpha
            alpha_t[alpha_t < 0] = 0      
            # update arrays
            C[niter] = 1/lambda_t
            alpha[niter] = alpha_t
            beta0[niter] = mu_t/lambda_t
            
            # update objective values
            t1 = alpha_t @ Q @ alpha_t
            t2 = alpha_t.sum()
            obj_svm[niter] = t1 / (2*lambda_t) - t2
            obj_dink[niter] = np.sqrt(np.abs(t1)) - lambda_t / 2 * t2
            
            Es[niter] = E
            Ls[niter] = L
            Rs[niter] = R
            
            # check for singular U matrix
            if len(U) > 0 and np.abs(np.diag(np.atleast_2d(U))).min() < 1e-5:
                lambda_t = lambda_min
                if verbose:
                    print("Singular matrix encountered... terminating path\n")
                
            
        t_path = default_timer() - s
        
        # final event
        events.append('End of Path')
        
        # truncate and set attributes
        self.C = C[:niter+1]
        self.dalpha = dalphas[:niter+1]
        self.dmu = dmus[:niter+1]
        self.alpha = alpha[:niter+1]
        self.beta0 = beta0[:niter+1]
        self.E = Es[:niter+1]
        self.L = Ls[:niter+1]
        self.R = Rs[:niter+1]
        self.obj_svm = obj_svm[:niter+1]
        self.obj_dink = obj_dink[:niter+1]
        self.events = events
        if not hasattr(self, 'lambda_final'): self.lambda_final = 1/self.C[-1]
        
        if verbose:
            print('  Initialization time: %2.3f' % t_init)
            print('Path computation time: %2.3f' % t_path)
            print('           Total time: %2.3f' % (t_init + t_path))
            
    # method for getting alpha for a given C value
    def get_alpha(self, C):
        # convert C to lambda
        lam = 1 / C
        
        # if separated and lam < lambda_final, return last solution in path
        if lam < self.lambda_final and self.separated:
            return self.alpha[-1]    
        # throw error if not separated but lam < lambda_final
        elif lam < self.lambda_final: 
            raise ValueError("Requested solution at C value %f unavailable. "
                             "Please specify a C value smaller than %.2e" %
                             (C, 1/self.lambda_final))
        # return initial solution if C smaller than initial value 
        elif lam > 1 / self.C[0]:
            return self.alpha[0]
        
        # otherwise compute using dalpha
        # find breakpoint segment
        ind = np.where(self.C < C)[0][-1] # closest value below C
        dlambda = 1 / C - 1 / self.C[ind]
        dalpha = self.dalpha[ind]
        
        alpha_new = self.alpha[ind] + dlambda * dalpha
        
        return alpha_new

def init_path(Q, y, eps=1e-6, aux_init=False, verbose=False):
    """Compute the initial solution for the SVM regularization path
    
    Parameters
    ----------
    Q : array_like
        The signed kernel matrix [y_i y_j K(x_i, x_j)].
        
    y : array_like
        The array of class labels, assumed to be in {-1, 1}.
        
    eps : float, optional
        Tolerance for determining whether or not a point is on the margin of 
        the SVM. Points with eps < alpha < 1 - eps are assumed to be marginal
        support vectors. The default is 1e-6.
        
    aux_init : bool, optional
        Boolean flag for specifying whether or not to use an alternative 
        initialization method in which artificial data points are added to
        enforce equal number of class labels in the augmented dataset, i.e.,
        n_p = n_n. This initialization avoids relying on an external solver and
        should only be used when a solver is not available or the solver cannot
        optimize the problem due to numerical issues. The default is False.

    Returns
    -------
    lambda0 : float
        The initial parameter on the path, lambda0 = 1 / C0.
        
    mu0 : float
        The intercept term scaled by lambda0, mu0 = lambda0 * beta0.
        
    alpha : array_like
        The dual coefficient solution corresponding to lambda0.
        
    E : array_like
        A boolean array of length n where index i is true if sample i is in the
        set of marginal support vectors.
        
    L : array_like
        A boolean array of length n where index i is true if sample i is in the
        set of non-marginal support vectors.
        
    R : array_like
        A boolean array of length n where index i is true if sample i is not a
        support vector.
    """     
    # data attributes
    n = len(y)
    n_p = int(y.sum() + n) // 2 # won't work with gurobipy without casting
    n_n = n - n_p
    
    pos_ind = np.where(y == 1)[0]
    neg_ind = np.where(y == -1)[0]
    
    # equal # of samples between classes
    if y.sum() == 0:
        # all samples are support vectors => alpha = 1
        alpha = np.ones(n)
        
        # w'x evaluated at all points
        f = y * Q.dot(alpha)
        
        # find most positive w'x for positive points
        fp = f[pos_ind]
        imax = fp.argmax()
        fp_max = f[pos_ind[imax]] # index relative to sample
        
        # find most negative w'x for negative points
        fn = f[neg_ind]
        imin = fn.argmin()
        fn_min = f[neg_ind[imin]]
        
        # lambda0 and beta0
        lambda0 = (fp_max - fn_min) / 2
        beta0 = -(fp_max + fn_min) / (fp_max - fn_min)
        
        # initialize E, L, R sets
        E = np.zeros(n,dtype=bool)
        L = np.ones(n,dtype=bool)
        R = np.zeros(n,dtype=bool)
        E_ind = [neg_ind[imin], pos_ind[imax]]
        E[E_ind] = 1
        L[E_ind] = 0
        
    # n_p > n_n
    else:
        # set up gurobi problem for finding initial alpha
        m = gp.Model("init")
        a = m.addMVar(n_p, lb=0, ub=1, vtype='C', name="alpha")
        
        # unsigned kernel submatrix (same as signed at pos_ind)
        Q_pos = Q[np.ix_(pos_ind,pos_ind)]
        # linear term coefficient vector (taking negative removes y_i y_j)
        c = -Q[np.ix_(pos_ind,neg_ind)].sum(axis=1)
        
        #  set constraint
        m.addConstr(a.sum() == n_n)
        
        # set objective
        m.setObjective(a @ Q_pos @ a - 2 * c @ a, GRB.MINIMIZE)
        
        # solve and get solution
        if not verbose: m.setParam("OutputFlag", 0)
        m.setParam('BarConvTol', 1e-11)
        m.setParam('PSDTol', 2e-5) # shouldn't be having psd issues
        try:
            m.optimize()
        except gp.GurobiError as e:
            # catching a psd error that shouldn't be occurring, happens with
            # polynomial kernel with large number of covariates
            if e.errno == 10020:
                Q_pos += np.diag(1e-12*np.ones(len(Q_pos)))
                m.setObjective(a @ Q_pos @ a - 2 * c @ a, GRB.MINIMIZE)
                m.optimize()
            else:
                raise e
        
        # throw warning if model cannot be solved optimally (usually a problem
        # with BarConvTol being too small)
        if m.Status == 13 and verbose:
            warn('Unable to solve initial problem optimally, using suboptimal '
                 'solution')
        
        # initialize alpha and truncate values to 0 and 1
        alpha = np.ones(n)
        alpha[pos_ind] = np.array(m.x)
        alpha[alpha >= 1 - eps] = 1
        alpha[alpha <= eps] = 0
        
        # indices corresponding to marginal samples (0 < alpha < 1)
        pmarg_ind = np.where((alpha != 1) & (alpha != 0))[0]
        # indices corresponding to positive support vectors (alpha = 1)
        psv_ind = np.where((alpha == 1) & (y == 1))[0]
        # indices corresponding non-support vectors (alpha = 0)
        pzero_ind = np.where((alpha == 0) & (y == 1))[0]
        
        # non-empty elbow for positive samples
        if len(pmarg_ind) > 0:
            # w'x over positive marginal samples
            fp = y[pmarg_ind] * Q[pmarg_ind].dot(alpha)
            # take the mean for numerical stability, though Hastie uses the 
            # max value instead. Not clear which would be better
            fp_max = fp.mean() # max is notation used in paper
            
        # empty elbow, go by positive sv nearest its margin
        else:
            fp = y[psv_ind] * Q[psv_ind].dot(alpha)
            imax = fp.argmax()
            fp_max = fp[imax]
        
        # negative sample on the margin      
        fn = y[neg_ind] * Q[neg_ind].dot(alpha)
        imin = fn.argmin()
        fn_min = fn[imin]
        
        # lambda0 and beta0
        lambda0 = (fp_max - fn_min) / 2
        beta0 = -(fp_max + fn_min) / (fp_max - fn_min)
        
        # partition index set into Elbow, Left, and Right
        E = np.zeros(n,dtype=bool)
        L = np.ones(n,dtype=bool)
        R = np.zeros(n,dtype=bool)
        
        E_ind = np.append(neg_ind[imin], pmarg_ind) if len(pmarg_ind) > 0 else\
                [neg_ind[imin], psv_ind[imax]]
        R_ind = pzero_ind
        E[E_ind] = 1
        L[np.concatenate((E_ind,R_ind))] = 0
        R[R_ind] = 1
        
    # make this a class eventually
    return (lambda0, lambda0*beta0, alpha, E, L, R)

# function for updating Cholesky factorization of Z'QZ
def update_cholesky(U, Q, y, q_new, y_new):
    """Update the current Cholesky factorization for when a new point is added
        
    Parameters
    ----------
    U : array_like
        Upper-triangular matrix corresponding to the current Cholesky 
        decomposition of the modified signed kernel sub-matrix, Z'QZ.
        
    Q : array_like
        The sub-matrix [y_i y_j K(x_i, x_j)] indexed by points on the 
        margin before updating. Equivalently, the upper-left diagonal 
        entry of the KKT matrix.
        
    y : array_like
        The vector of class labels corresponding to points on the
        margin before updating.
        
    q_new : array_like
        The row/column of the matrix [y_i y_k K(x_i,x_j)] to be added 
        to Q.
        
    y_new : scalar
        The class label corresponding to q_new.
             
    Returns
    -------
    U_new : array_like
        The updated Cholesky factorization matrix. Note that U is 
        upper-triangular by definition.
    """
    # make sure y has a length
    y = np.atleast_1d(y)
    # special cases 
    if len(y) == 0: # Q_new is one-dimensional
        U_new = np.sqrt(q_new)
    elif len(y) == 1: # (Z'Q_new Z is one-dimensional)
        U_new = np.sqrt(Q + q_new[-1] - 2 * y * y_new * q_new[0])
    else:
        # compute old null-matrix for y'
        Z = np.vstack((-y[0] * y[1:], np.eye(len(y) - 1)))
        
        # split q_new into off-diagonal and diagonal terms
        q, sigma = q_new[:-1], q_new[-1]
        
        # solve for u (see Sentelle 2012 for details)
        rhs = -y[0] * y_new * Z.T.dot(Q[0]) + Z.T.dot(q)
        
        # solve linear system
        u = solve_triangular(np.atleast_2d(U).T, rhs, lower=True)
        
        # solve for rho
        rho2 = Q[0,0] - 2 * y[0] * y_new * q[0] + sigma - u.dot(u)
        rho = np.sqrt(rho2) if rho2 >= 0 else np.nan
        
        # form new U matrix
        U_new = np.block([[U, u[:,np.newaxis]],
                          [np.zeros(len(U)), rho]])
        # guard agains numerical issues
        # U_new[np.tril_indices_from(U_new,k=-1)] = 0
        
    # return 
    return U_new

# function for downdating Cholesky factorization of Z'QZ
def downdate_cholesky(U, y, ind):
    """Update the current Cholesky factorization for when a new point is added
        
    Parameters
    ----------
    U : array_like
        Upper-triangular matrix corresponding to the current Cholesky 
        decomposition of the modified signed kernel sub-matrix, Z'QZ.
        
    y : array_like
        The vector of class labels corresponding to points on the
        margin before updating.
        
    ind : scalar
        Index of the row/column to be removed from the sub-matrix Q.
         
    Returns
    -------
    U_new : array_like
        The updated Cholesky factorization matrix. Note that U is 
        upper-triangular by definition.
    """ 
    # return empty matrix when U is one-dimensional
    if len(np.atleast_1d(U)) == 1:
        U_new = np.array([])
    # when ind does not correspond to the first row/column of Q
    elif ind > 0:
        # delete ind-1 column of U (corresponding to deleting ind from Q)
        U_new = np.delete(U, ind-1, 1)
        # perform Givens rotations to convert matrix back to upper-triangular
        for i in range(ind-1, len(U_new)-1):
            U_new[i:i+2] = givens(U_new[i,i], U_new[i+1,i]) @ U_new[i:i+2]
            
        # truncate
        U_new = U_new[:-1]
        # guard against numerical issues (probably not needed)
        # U_new[np.tril_indices_from(U_new,k=-1)] = 0
        
    # first index deleted    
    else:
        # updated null space matrix concatenated with Y
        Z = np.vstack((np.append(-y[1] * y[2:], 1), 
                       np.hstack((np.eye(len(y)-2), 
                                  np.zeros((len(y)-2,1)))
                                 )
                       ))
        # update R
        U_new = U @ Z
        
        # perform Givens rotations to transform back to upper-triangular
        for i in range(len(U_new)-1):
            U_new[i:i+2] = givens(U_new[i,i], U_new[i+1,i]) @ U_new[i:i+2]
            
        # truncate
        U_new = U_new[:-1,:-1]
        # guard against numerical issues (probably not needed)
        # U_new[np.tril_indices_from(U_new,k=-1)] = 0
        
    # return (need to squeeze for 2x2 -> 1x1)
    return np.atleast_1d(U_new.squeeze())

# function for computing the search direction by solving a linear system
def compute_dirs(U, Q, y):
    """Solve the linear system for computing directions via the null space method
        
    Parameters
    ----------
    U : array_like
        Upper-triangular matrix corresponding to the current Cholesky 
        decomposition of the modified signed kernel sub-matrix, Z'QZ.
        
    Q : array_like
        The sub-matrix [y_i y_j K(x_i,x_j)] indexed by points on the 
        margin of the current SVM solution. Equivalently, the upper-left
        diagonal entry of the KKT matrix.
        
    y : array_like
        The vector of class labels corresponding to points on the
        margin of the current SVM solution
             
    Returns
    -------
    dalpha : array_like
        The change in alpha with respect to lambda.
        
    dmu : scalar
        The change in mu with respect to lambda.
    """
    # special case for len(y) <= 1
    if len(y) <= 1:
        dalpha = 0
        dmu = y[0]
    else:
        # null space of y' (row vector of y)
        Z = np.vstack((-y[0] * y[1:], np.eye(len(y) - 1)))
        
        # see Numerical Optimization (Wright 2006) for details
        
        # compute da_y through y'Y da_y = y'e_1 da_y = 0 => da_y = 0
        
        # next compute da_z through U'U da_z = Z'(1 - y dmu) = Z'1
        rhs = Z.T.dot(np.ones(len(y)))
        da_z = cho_solve((np.atleast_2d(U), False), rhs)
        
        # dalpha = Z da_z + Y da_y = Z da_z
        dalpha = Z.dot(da_z)
        
        # dmu = y_1 * (1 - Q_1' dalpha)
        dmu = y[0] * (1 - Q[0].dot(dalpha))
        
    return (dalpha, dmu)

# function for computing the Givens rotation matrix
def givens(x, y):
    # comput c and s (see Wikipedia entry for Givens rotation)
    c, s = drotg(x,y)
    
    return np.array([[c, s], [-np.conj(s), c]])
