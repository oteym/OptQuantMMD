# =============================================================================
# CODE TO BATCH SELECT SAMPLES USING KSD OPTIMALITY CRITERION.
# BASED ON PAPER "OPTIMAL QUANTISATION OF PROBABILITY MEASURES USING MAXIMUM
# MEAN DISCREPANCY" BY TEYMUR, GORHAM, RIABIZ & OATES (AISTATS 2021)
#
# USES PACKAGED OPIMISATION PACKAGES THAT REQUIRE ADDITIONAL LICENCES
# FOR GREEDY OPTIMISATION: GUROBI (AVAILABLE FREE TO ACADEMIC USERS)
# FOR SEMIDEFINITE RELAXATION: MOSEK (AVAILABLE FREE TO ACADEMIC USERS)
# OPEN-SOURCE PYTHON-NATIVE OPTIMISERS AVAILABLE AND CAN EASILY BE SUBSTITUTED
#
# EXAMPLE MCMC CHAIN LOADED AND OPTIMAL SELECTION PERFORMED USING KSD (WITH
# GRADIENTS); MMD OPTIMISATION REQUIRES ANALYTICAL EXPRESSION FOR TARGET.
# CORRESPONDS TO ALGORITHM 3 IN TEYMUR ET AL.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from timeit import default_timer as timer
from gurobipy import GRB
import mosek.fusion as ms

#############################################################################

### HELPER FUNCTIONS:

# MEDIAN HEURISTIC FOR KERNEL LENGTHSCALE
def median_heuristic(X,subset=0):
    dists = []
    if subset==0:
        subset = X.shape[0]
        subset_indices = [*range(subset)]
    elif subset>0:
        subset_indices = np.random.choice(X.shape[0],subset,replace=False)
    tmp1 = np.tile(X[subset_indices,:],[subset,1]) - np.repeat(X[subset_indices,:],subset,axis=0) 
    tmp2 = np.sum(tmp1 * tmp1,axis=1)
    med = np.sqrt(np.median(tmp2) / 2)
    return med

# MULTIQUADRIC STEIN KERNEL MATRIX
def mskm(X_A,X_B,score_A,score_B,l):
    n_A = X_A.shape[0]
    n_B = X_B.shape[0]
    d = X_A.shape[1]
    if X_B.shape[1] != d: raise ValueError("dimensions incompatible")
    if callable(score_A) == True and callable(score_B) == True:
        score_A_vec = [score_A(X_A[i,:]) for i in range(0,n_A)]
        score_B_vec = [score_B(X_B[i,:]) for i in range(0,n_B)]    
    elif callable(score_A) == False and callable(score_B) == False:
        score_A_vec = score_A
        score_B_vec = score_B
    else: raise ValueError("score inputs of mixed type")
    tmp0 = d/l/l;
    tmp1 = np.sum(np.tile(score_A_vec,[n_B,1]) * np.repeat(score_B_vec,n_A,axis=0),axis=1)
    tmp2 = np.tile(X_A,[n_B,1]) - np.repeat(X_B,n_A,axis=0)
    tmp3 = np.tile(score_A_vec,[n_B,1]) - np.repeat(score_B_vec,n_A,axis=0)
    tmp4 = tmp2/(l**2)
    tmp6 = 1 + np.sum(tmp2 * tmp4, axis=1)
    tmp5 = -3 * np.sum(tmp4 * tmp4, axis=1) / (tmp6**(5/2)) +\
        (tmp0 + np.sum(tmp3 * tmp4, axis=1)) / (tmp6**(3/2)) +\
        tmp1 / (tmp6**(1/2))
    K = np.reshape(tmp5,(n_B,n_A))
    return K

# DIAGONAL OF MULTIQUADRIC STEIN KERNEL MATRIX (WITHOUT CALCULATING ENTIRE MATRIX)
def mskd(X,score,l):
    n = X.shape[0]
    d = X.shape[1]
    if callable(score) == True:
        score_vec = [score(X[i,:]) for i in range(0,n)]
    elif callable(score) == False:
        score_vec = score
    else: raise ValueError("score inputs of mixed type")
    tmp0 = d/l/l;
    tmp1 = np.sum(score_vec * score_vec,axis=1)
    k_diag = tmp0 + tmp1
    return k_diag

# PLOTTING FUNCTION
def scatter_plot(z,z_subset,annotate_toggle=0,lims=0,selection_toggle=1): #R
    fig = plt.figure(figsize=(5, 5))
    plt.gca().set_aspect('equal', adjustable='box')  
    plt.plot(z[:,0],z[:,1],'.',color='b',)
    if selection_toggle == 1:
        plt.plot(z_subset[:,0],z_subset[:,1],'.',color='r',markersize=10)
    if annotate_toggle==1:
        for i in range(z_subset.shape[0]):
            plt.annotate(i+1, (z_subset[i,0], z_subset[i,1]),fontsize=16)
    if lims!=0:
        plt.xlim((lims[1][0]),(lims[1][1]))
        plt.ylim((lims[0][0]),(lims[0][1]))
    plt.show()
    return fig

#############################################################################

### MAIN FUNCTION:

def point_selection_KSD(X,n,M,b,s,l,score,batch_replacement=1,time_limit=2,sdr=0,seed=0,tt=1):
    
    ### TIMER
    start = timer()
    np.random.seed(seed)

    ### CHECK INPUT COMPBATIBILITIES
    if X.shape[0] < n: raise ValueError("requested dataset size larger than given dataset")
    if n == 0: n = X.shape[0]
    elif b > n: raise ValueError("batch size larger than dataset")

    ### SET UP ARRAYS
    indices = []
    remaining = [*range(n)]
    m = int(np.ceil(M//s)) # number of iterations. makes up at least M
    if m*b > n and batch_replacement == 0: raise ValueError("not enough points to do non-replacement batching")

    ### DEFINE KERNEL SUBROUTINE (USES MULTIQUADRIC)
    ker_mat_call = mskm
    ker_diag_call = mskd
    
    ### IF CONDIITION MET, CALCULATE KERNEL MATRIX ONCE ONLY
    if s > 1 and b*b*m >= n*n:
        ker_mat_full = ker_mat_call(X[0:n,:],X[0:n,:],score[0:n,:],score[0:n,:],l)
    elif s == 1 and b*b*m >=  n*n:
        ker_diag_full = ker_diag_call(X[0:n,:],score[0:n,:],l)
        
    ### DECLARE ARRAYS
    f = np.full(b,0.0)
    lap = np.full(m,0.0)
    running_samps = np.full(m,0.0)
    running_ksd = np.full(m,0.0)
    loop_count = 0
    
    ### MAIN LOOP
    for j in range(m):
        
        print(j)
        
        ### COUNTER
        loop_count = loop_count+1

        ### DETERMINE MINIBATCH INDICES (ALL OPTIONS RESULT IN BI A VECTOR OF LENGTH b)
        if n == b: BI = np.array([*range(n)])
        elif batch_replacement == 1: BI = np.random.choice(n,b,replace=False)
        elif batch_replacement == 0: BI = np.random.choice(remaining,b,replace=False)  # batch indices
        elif batch_replacement == 2: BI = np.array([*range(j*b,(j+1)*b)]) # systematic batch (streaming) for comparison with wilson

        c = len(indices)
        
        ### USE PRE-CALCULATED COMPLETE KERNEL MATRIX OR CALCULATE BATCH-BY-BATCH
        if s > 1 and b*b*m >= n*n:    
            ker_mat = ker_mat_full[BI[:,None],BI]
            f = np.sum(ker_mat_full[BI[:,None],indices],axis=1)
        elif s > 1 and b*b*m < n*n:    
            ker_mat = ker_mat_call(X[BI,:],X[BI,:],score[BI,:],score[BI,:],l)
            f = np.sum(ker_mat_call(X[BI,:],X[indices,:],score[BI,:],score[indices,:],l),axis=0)
        elif s == 1 and b*b*m >= n*n:
            ker_diag = ker_diag_full[BI]
            f = np.sum(ker_mat_call(X[BI,:],X[indices,:],score[BI,:],score[indices,:],l),axis=0)
        elif s == 1 and b*b*m < n*n:
            ker_diag = ker_diag_call(X[BI,:],score[BI,:],l)
            f = np.sum(ker_mat_call(X[BI,:],X[indices,:],score[BI,:],score[indices,:],l),axis=0)
            
        ### GREEDY OPTIMISATION USING GUROBI:
        if sdr == 0:
    
            if s == 1:
                if j == 0:
                    x_int = [np.nanargmin(ker_diag)]
                elif j > 0:
                    vals = 0.5*ker_diag + f
                    x_int = [np.nanargmin(vals)]
            elif s > 1:
                gurobi_model = gp.Model("test")
                x = gurobi_model.addMVar(b, vtype=GRB.BINARY, name="x")
                gurobi_model.setObjective(x @ (0.5 * ker_mat) @ x + f @ x, GRB.MINIMIZE)
                gurobi_model.addConstr(x.sum() == s, name="c")
                gurobi_model.Params.OutputFlag = 0
                if time_limit!=0:
                    gurobi_model.Params.TimeLimit = time_limit
                gurobi_model.optimize()
                x_int = np.nonzero(x.X>0.5)[0].tolist()
                        
            ### CALCULATE KSD
            if j == 0: r = 0
            else: r = running_ksd[j-1]
            ksd_sequential = np.sqrt( (c+s)**(-2) * (  (r*c)**2 +\
                np.sum(ker_mat_call(X[BI[x_int],:],X[BI[x_int],:],score[BI[x_int],:],score[BI[x_int],:],l)) +\
                2 * np.sum(ker_mat_call(X[BI[x_int],:],X[indices,:],score[BI[x_int],:],score[indices,:],l) )   ))

        ### SEMI-DEFINITE RELAXATION USING MOSEK:
        elif sdr == 1:
            
            A = np.zeros((b+1,b+1))
            A[1:b+1,1:b+1] = ker_mat
            A[0,1:b+1] = np.ones(b) @ ker_mat + 2*f
            A[1:b+1,0] = A[0,1:b+1]
            A_mosek = ms.Matrix.dense(A)
            
            B = np.zeros((b+1,b+1))
            B[0,1:b+1] = 1/2
            B[1:b+1,0] = 1/2
            B[0,0] = 0
            B_mosek = ms.Matrix.dense(B)
        
            mosek_model = ms.Model("sdr")
            M = mosek_model.variable(ms.Domain.inPSDCone(b+1))
            mosek_model.objective(ms.ObjectiveSense.Minimize, ms.Expr.dot(A_mosek, M))
            for i in range(b+1): mosek_model.constraint(M.index([i,i]), ms.Domain.equalsTo(1))
            mosek_model.constraint(ms.Expr.dot(B_mosek,M), ms.Domain.equalsTo(2*s-b))
            mosek_model.solve()
            M = np.reshape(M.level(), (b+1,b+1))
            U = np.linalg.cholesky(M[1:b+1,1:b+1]) # cholesky decomp of V
            r = np.random.normal(0,1,(b,tt)) #no need to normalise
            x_int_candidates = np.argsort(np.dot(r.transpose(),U))[:,:s]
            
            ksd_sequential = np.zeros(tt)
            for i in range(tt):
                if j == 0: r = 0
                else: r = running_ksd[j-1]
                x_int = x_int_candidates[i].tolist()
                ksd_sequential[i] = np.sqrt( (c+s)**(-2) * (  (r*c)**2 +\
                    np.sum(ker_mat_call(X[BI[x_int],:],X[BI[x_int],:],score[BI[x_int],:],score[BI[x_int],:],l)) +\
                    2 * np.sum(ker_mat_call(X[BI[x_int],:],X[indices,:],score[BI[x_int],:],score[indices,:],l) )   ))
            x_int = x_int_candidates[np.argmin(ksd_sequential)].tolist()
            ksd_sequential = ksd_sequential[np.argmin(ksd_sequential)]

        ### APPEND NEW SAMPLES TO COLLECTION
        for i in range(len(x_int)): indices.append(BI[x_int[i]])
        if batch_replacement == 0: remaining = np.setdiff1d([*range(n)],BI)

        ### TIMER    
        lap[j] = timer()-start
        running_samps[j] = 2*(j+1)*b
        running_ksd[j] = ksd_sequential
    
    return(indices,np.concatenate([running_samps.reshape(-1,1),lap.reshape(-1,1),running_ksd.reshape(-1,1)],1))

#############################################################################

### LOAD TEST DATA
data = np.load('data.npz')
points = data['points']
scores = data['scores']

### ESTIMATE KERNEL LENGTHSCALE USING MEDIAN HEURISTIC
np.random.seed(1)
l_ksd = median_heuristic(points)

### CALL MAIN ROUTINE
# Input arguments: X = data, \
#                  n = total number of data to consider (first n rows of X), \
#                  M = total number of points to collect, \
#                  b = mini-batch size, \
#                  s = number of points to select simultaneously, \
#                  l = kernel lengthscale
#                  dX = gradient of model at sample locations, \
#                  batch_replacement = 0: random without replacement, 1: random with replacement, 2: streaming mode (systematic batching)
#                  sdr = 0: greedy optimisation with gurobi, 2: semi definite relaxation with mosek
#                  time_limit = gurobi optimiser input to fix computation time, \
#                  seed = random seed for reproducibility, \
#                  tt = number of random projections to perform in SDR mode; the best performing is chosen   

out,timing = point_selection_KSD(points,1000,40,1000,20,l_ksd,scores,batch_replacement=1,sdr=0,time_limit=2,seed=10,tt=200)

# Outputs:         'out' returns array of indices of selected samples/
#                  'timing' returns array of cumulative no of samples collected (column 1), \
#                       wall-clock time (column 2), and KSD of the empirical measure formed \
#                       of the samples so far collected (column 3)

### PLOT
scatter_plot(points,points[out,:],0,0,1)
