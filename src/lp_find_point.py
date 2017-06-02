"""
Created on Tue March 7

@author: stonneau
"""

import os, sys
sys.path.insert(0, './tools')

from polytope_conversion_utils import crossMatrix
from numpy import array, zeros, ones, sqrt, cross, identity, asmatrix, empty
from numpy.random import rand, uniform
from CWC_methods import compute_CWC, is_stable
from lp_dynamic_eq import dynamic_equilibrium_lp
from numpy.linalg import norm

NUMBER_TYPE = 'float'  # 'float' or 'fraction'
__EPS =  2.39847622652e-4
__ACC_MARGIN =  0
#~ __EPS =  0

#~ from scipy.optimize import linprog
from pinocchio_inv_dyn.optimization import solver_LP_abstract

def __compute_K_c_kin(K_c,Kin):
    if Kin == None:
        return K_c
    K_kin = Kin[0]
    K_c_kin = zeros((K_c.shape[0] + K_kin.shape[0], K_c.shape[1]) )
    #~ print "k_c shape ", K_c.shape
    #~ print "K_kin shape ", K_kin.shape
    #~ print "K_kin", K_kin
    #~ print "K_c_kin shape ", K_c_kin.shape
    K_c_kin [:K_c.shape[0], :] = K_c
    K_c_kin [K_c.shape[0]:,:] = K_kin
    return K_c_kin
    
def __compute_k_c_kin(k_c,Kin):
    if Kin == None:
        return k_c
    k_kin = Kin[1]
    #~ print "k_kin", k_kin
    k_c_kin = zeros(k_c.shape[0] + k_kin.shape[0])
    k_c_kin[:k_c.shape[0]] = k_c[:]
    k_c_kin[k_c.shape[0]:] = -k_kin[:]
    return k_c_kin


def __compute_K_1(K, ones_range):
    K_1 = zeros((K.shape[0],4))
    #~ K_1 = ones((K.shape[0],4)) * -1
    K_1[ones_range[0]:ones_range[1],-1] = ones(ones_range[1]-ones_range[0])
    #~ K_1 = ones((K.shape[0],4))
    #~ K_1[ones_range[0]:ones_range[1],-1] = zeros(ones_range[1]-ones_range[0])
    K_1[:,:3] = K[:]
    return K_1

def __normalize(A,b):
    for i in range (A.shape[0]):
        n_A = norm(A[i,:])
        if(n_A != 0.):
            A[i,:] = A[i,:] / n_A
            b[i] = b[i] / n_A
    return A, b
 
def lp_ineq_4D(K,k, ones_range = None):
    if(ones_range == None):
        ones_range = (0, K.shape[0])
    K, k =  __normalize(K, k)
    K_1 = __compute_K_1(K, ones_range)
    cost = array([0.,0.,0.,-1.])
    lb =  array([-100000000. for _ in range(4)]);
    ub =  array([ 100000000. for _ in range(4)])
    Alb = array([-100000000. for _ in range(k.shape[0])])
    #~ K_1, k =  __normalize(K_1, k)
    solver = solver_LP_abstract.getNewSolver('qpoases', "dyn_eq", maxIter=10000, maxTime=10000.0, useWarmStart=False, verb=0)
    (status, res, rest) = solver.solve(cost, lb = lb, ub = ub, A_in=K_1, Alb=Alb, Aub=-k, A_eq=None, b=None)
    
    #problem solved or unfeasible
    status_ok = status== solver_LP_abstract.LP_status.OPTIMAL
    p_solved = status_ok and res[3] >= 0.
    return status_ok, p_solved , res
    
    
from scipy.linalg import block_diag, norm
from numpy import array, arange, zeros, ones, identity, vstack, hstack, append, sqrt, square, sum    
    
from scipy.optimize import minimize
def qp_ineq_4D(c_ref, K,k, ones_range = None):
    #~ K_1 = __compute_K_1(K, ones_range)
    #~ K_1, k =  __normalize(K_1, k)
    K, k =  __normalize(K, k)
    a_c_ref = array(c_ref)
    fun = lambda x: sum((x-a_c_ref)**2)
    if ones_range != None:
		K = __compute_K_1(K, ones_range)
		fun = lambda x: sum((x-a_c_ref)**2) * 10 + x[3]
		a_c_ref = array(c_ref+[0])
    #in slsqp constraint is Ax + b >= 0
    # we have Kx + k <=0 
    
		
    cons = ({'type': 'ineq',
            'fun' : lambda x: -(K.dot(x)+k)})
    res = minimize(fun, a_c_ref, constraints=cons, method='SLSQP', options={'ftol': 1e-06, 'maxiter' : 500})
    return res

# ********************************************************
# ********************************************************
# ********************* CWC METHODS **********************
# ********************************************************
# ********************************************************
 
#********* BEGIN find_intersection_c ********************
def __compute_H(H1, H2):
    assert H1.shape[1] == H2.shape[1], "matrix do not have the same dimension"
    H_tot = zeros((H1.shape[0]+ H2.shape[0],H1.shape[1]));
    H_tot[0:H1.shape[0],:] = H1[:]
    H_tot[H1.shape[0]:,:] = H2[:]
    return H_tot
    
def __compute_K_c(H, w1):
    H_w2 = H[:,3:]
    w1_cross = crossMatrix(w1)
    return H_w2.dot(-w1_cross)        
    
def __compute_k_c(H, w1):
    H_w1 = H[:,0:3]
    k = zeros(H.shape[0])
    k[:] = H_w1.dot(w1)
    return k

# Find a COM lying in a polytope for a given com acceleration, maximizing the distance
# to bounds, assuming dL = 0
# Find x=[c_x, c_y, c_z, s]
# min -s  ([0,0,0,-1]* x)
# K_1_c x + k_c<= 0
# K_1_c and k_c are  computed as follows, given that w1 is the first three lines of w: m * (ddc - g)
# H = [H_w1 ; H_w2]
# H_w2 * c_cross *  w1 + H_w1 * w1 <= 0
# H_w2 * -w1_cross * c + H_w1 * w1 <= 0
# K_c * c + k_c <= 0
# We then add kinematic constraints:
# Kin *c <= kin_c
# => [K_c; Kin]^T c + [k_c; -kin_c]^T <= 0
# => K_c_kin c + k_c_kin <= 0
# then adding s    
# [K_c_kin, 1_n] *  x + k_c_kin<= 0
#  K_1_c x + k_c_kin<= 0
#  \param H CWC for a contact phase
#  \param ddc selected acceleration
#  \param Kin polytope of kinematic constraints on the COM position. If None, then the COM is not constrained
#  \param mu friction coefficient
#  \param g_vec gravity acceleration
#  \return the solver status, whether the point satisfies the constraints, and the closest point that satisfies them
def find_valid_c_cwc(H, ddc, Kin = None, only_max_kin = False, only_max_dyn = False, m = 54., g_vec=array([0.,0.,-9.81])):
    w1 = m * (ddc - g_vec)
    K_c = __compute_K_c(H, w1)
    k_c = __compute_k_c(H, w1)
    K_c_kin = __compute_K_c_kin(K_c,Kin)
    k_c_kin = __compute_k_c_kin(k_c,Kin)
    one_range = None
    if(only_max_kin):
        one_range=(K_c_kin.shape[0] - K_c.shape[0],K_c_kin.shape[0])
    elif only_max_dyn:
        one_range=(0, K_c.shape[0])
    return lp_ineq_4D(K_c_kin,k_c_kin, one_range)

# Find a COM lying in a polytope for a given com acceleration, minimizing the distance to a reference point, assuming dL = 0
# Find x=[c_x, c_y, c_z, s]
# min ||c_ref - x||^2 - s
# K_1_c x + k_c<= 0
# K_1_c and k_c are  computed as follows, given that w1 is the first three lines of w: m * (ddc - g)
# H = [H_w1 ; H_w2]
# H_w2 * c_cross *  w1 + H_w1 * w1 <= 0
# H_w2 * -w1_cross * c + H_w1 * w1 <= 0
# K_c * c + k_c <= 0
# We then add kinematic constraints:
# Kin *c <= kin_c
# => [K_c; Kin]^T c + [k_c; -kin_c]^T <= 0
# => K_c_kin c + k_c_kin <= 0
# then adding s    
# [K_c_kin, 1_n] *  x + k_c_kin<= 0
#  K_1_c x + k_c_kin<= 0
#  \param H CWC for a contact phase
#  \param ddc selected acceleration
#  \param Kin polytope of kinematic constraints on the COM position. If None, then the COM is not constrained
#  \param mu friction coefficient
#  \param g_vec gravity acceleration
#  \return the solver status, whether the point satisfies the constraints, and the closest point that satisfies them
def find_valid_c_cwc_qp(H, c_ref, Kin = None, ddc=[0.,0.,0.], m = 54., g_vec=array([0.,0.,-9.81])):
    w1 = m * (ddc - g_vec)
    K_c = __compute_K_c(H, w1)
    k_c = __compute_k_c(H, w1)
    one_range = None
    if Kin != None:
		K_c = __compute_K_c_kin(K_c,Kin)
		k_c = __compute_k_c_kin(k_c,Kin)
		one_range=(0, K_c.shape[0])		
    #~ one_range = None
    #~ if(only_max_kin):
        #~ one_range=(K_c_kin.shape[0] - K_c.shape[0],K_c_kin.shape[0])
    return qp_ineq_4D(c_ref, K_c,k_c, one_range)

#********* END find_intersection_c ********************
    
#********* BEGIN find_intersection_ddc ********************
def __compute_D(c):
    res = zeros((6,3));
    res[0:3,:] = identity(3)
    res[ 3:,:] = crossMatrix(c)
    return res

def __compute_d(c,g):
    res = zeros(6);
    res[0:3] = -g
    res[ 3:] = crossMatrix(c).dot(-g)
    return res

def __compute_K_ddc(m, H, c):
    D = __compute_D(c);
    return m*(H.dot(D))
    
def __compute_k_ddc(m, H, c, g):
    d = __compute_d(c,g);
    return m*(H.dot(d))


# Find a COM acceleration ddc lying in a polytope maximizing distance to bounds
# assuming dL = 0, and given a COM position c
# Find x=[ddc_x, ddc_y, ddc_z, s]
# min -s  ([0,0,0,-1]* x)
# K_1_ddc x + k_ddc<= 0
# K_1_ddc and k_ddc are  computed as follows:
# we have w = [w1 w2]^T in R^6 
# w1 =  m(ddc - g)
# w2 =  c_cross * m (ddc - g)
# we rewrite it as a function of ddc
# w =  m ([Id_{3} ; c_cross]^T ddc + [-g ; c_cross * -g]) = m(D * ddc + d)
# which gives 
# m*H*D ddc + mH * d < = 0
# K_ddc ddc +  k_ddc < = 0
# adding constraint of maximizing s:
#   [K 1 ]  * x + k_ddc < = 0
#   K_1_ddc * x + k_ddc < = 0
#  if no such point exists, returns the closest point to the constraints
#  \param H CWC for a contact phase
#  \param c selected COM position
#  \param mu friction coefficient
#  \param g_vec gravity acceleration
#  \return the solver status, whether the point satisfies the constraints, and the closest point that satisfies them
def find_valid_ddc_cwc(H, c, m = 54., g_vec=array([0.,0.,-9.81])):
    K_ddc = __compute_K_ddc(m, H, c)
    k_ddc   = __compute_k_ddc  (m, H, c, g_vec)
    return lp_ineq_4D(K_ddc,k_ddc)
    

#********* END find_intersection_ddc ********************

# Find a combination of c and ddc (assuming dL = 0) such that the generated wrench
# lies in the code. Achieves this by calling recursively find_valid_ddc and find_valid_c
# assuming dL = 0, and given a COM acceleration ddc
#  \param H CWC for a contact phase
#  \param max_iter maximum number of trials before giving up trying to find a solution
#  \param ddc initial guess for COM acceleration
#  \param mu friction coefficient
#  \param g_vec gravity acceleration
#  \return [(c,ddc), success, margin] where success is True if a solution was found and margin is the the minimum distance to the bounds found
def find_valid_c_ddc_cwc(H, max_iter = 5, ddc=array([0.,0.,0.]), m = 54.,  g_vec=array([0.,0.,-9.81])):
    current_iter = max_iter
    __ddc = ddc[:] 
    
    while(current_iter > 0):
        current_iter -= 1
        status, sol_found, wp_1 = find_valid_c_cwc(H, __ddc, m = m, g_vec = g_vec)
        if(status != 0):
            print "[ERROR] LP find_intersection_c is not feasible"
            return
        c= wp_1[0:3][:]
        
        margin = wp_1[3]    
        sol_found = sol_found and margin > __ACC_MARGIN     
        
        if(sol_found):
            print "FOUND DIRECTLY solution, (c / ddc) ", c , " " , __ddc, "margin", margin
            return [(c,__ddc), True, margin]
        if(True):
            print "no solution found for acceleration " , __ddc , " (margin ", margin, ") , try to find acceleration with best c", c
            status, sol_found, wp_1 = find_valid_ddc_cwc(H, c)
            if(not sol_found):            
                if(status != 0):
                    print "[ERROR] LP find_intersection_ddc is not feasible"
                print "no solution found for the two phases"
                return
            __ddc = wp_1[0:3][:]
            margin = wp_1[3]    
            print "best found acc this turn (margin), ", __ddc, margin
        else:
            print "directly found SOLUTION, ", margin
        if margin > __ACC_MARGIN:
            print "solution, (c / ddc) ", c , " " , __ddc, "margin", margin
            print "number of iterations required ", max_iter - current_iter
            return [(c,__ddc), True, margin]
    return [(c,__ddc), False, margin]
    
    
# ********************************************************
# ********************************************************
# ********************* LP METHODS **********************
# ********************************************************
# ********************************************************
    
# Find a combination of c and ddc (assuming dL = 0) such that the generated wrench
# lies in the code. Achieves this by by random sampling
#  \param P Contact points for a contact phase
#  \param N Contact normals for a contact phase
#  \param max_iter maximum number of trials before giving up trying to find a solution
#  \param mu friction coefficient
#  \param g_vec gravity acceleration
#  \return [(c,ddc), success, margin] where success is True if a solution was found and margin is the the minimum distance to the bounds found
def find_valid_c_ddc_random(P, N, Kin = None, bounds_c = [0.,1.,0.,1.,0.,1.], bounds_ddc = [-1.,1.,-1.,1.,-1.,1.], max_iter = 10000, m = 54., mu = 0.6, g_vec=array([0.,0.,-9.81])):
    c = None; ddc = None;
    for _ in range(max_iter):
        c = array([uniform(bounds_c[2*i], bounds_c[2*i+1]) for i in range(3)])
        if Kin == None or (Kin[0].dot(c)<=Kin[1]).all():
            for _ in range(max_iter):
                ddc = array([uniform(bounds_ddc[2*i], bounds_ddc[2*i+1]) for i in range(3)])
                res_lp, robustness =  dynamic_equilibrium_lp(c, ddc, P, N, mass = m, mu = mu)
                if robustness >= 0:
                    return [(c,ddc), True, robustness]
    #~ print "never found  a valid solution "
    return [(c,ddc), False, robustness]
    
# Find a c (assuming dL = 0 and ddc=0) such that the generated wrench
# lies in the code. Achieves this by by random sampling
#  \param P Contact points for a contact phase
#  \param N Contact normals for a contact phase
#  \param max_iter maximum number of trials before giving up trying to find a solution
#  \param mu friction coefficient
#  \param g_vec gravity acceleration
#  \return [(c,ddc), success, margin] where success is True if a solution was found and margin is the the minimum distance to the bounds found
def find_valid_c_random(P, N, Kin = None, bounds_c = [0.,1.,0.,1.,0.,1.], bounds_ddc = [-1.,1.,-1.,1.,-1.,1.], max_iter = 10000, m = 54., mu = 0.6, g_vec=array([0.,0.,-9.81]), no_eq = False):
    c = None; ddc = None;
    for i in range(max_iter):
        c = array([uniform(bounds_c[2*i], bounds_c[2*i+1]) for i in range(3)])
        ddc = array([0.,0.,0.])
        res_lp, robustness =  dynamic_equilibrium_lp(c, ddc, P, N, mass = m, mu = mu)
        if no_eq or robustness >= 0.01:
            #~ print "found a valid solution in ", i , "trials: ", (c,ddc), "robustness : ", robustness
            if Kin != None:
                #~ print "checking for boundaries, ", (Kin[0].dot(c)).T
                if(Kin[0].dot(c)<=Kin[1]).all():
                    #~ print "boundaries satisfied"
                    return [(c,ddc), True, robustness]
            else:
                return [(c,ddc), True, robustness]
    #~ print "never found  a valid solution "
    return [(c,ddc), False, robustness]
