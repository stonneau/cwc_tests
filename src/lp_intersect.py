"""
Created on Tue March 7

@author: stonneau
"""

import os, sys
sys.path.insert(0, './tools')

from polytope_conversion_utils import crossMatrix
from numpy import array, zeros, ones, sqrt, cross
NUMBER_TYPE = 'float'  # 'float' or 'fraction'

#~ g_vec = array([0,0,-9.81])

#~ from scipy.optimize import linprog
from pinocchio_inv_dyn.optimization import solver_LP_abstract
 
solver = solver_LP_abstract.getNewSolver('qpoases', "dyn_eq", maxIter=1000, maxTime=100.0, useWarmStart=True, verb=0)


def __compute_H(H1, H2):
	assert H1.shape[1] == H2.shape[1], "matrix do not have the same dimension"
	H_tot = zeros((H1.shape[0]+ H2.shape[0],H1.shape[1]));
	H_tot[0:H1.shape[0],:] = H1[:]
	H_tot[H1.shape[0]:,:] = H2[:]
	return H_tot
	
	
def __compute_K(H, w1):
	H_w1 = H[:,0:3]
	w1_cross = crossMatrix(w1)
	return -H_w1.dot(w1_cross)
	
def __compute_K_1(H, w1):
	K = __compute_K(H, w1)
	K_1 = ones((K.shape[0],4))
	K_1[:,:3] = K[:]
	return K_1
	
def __compute_k(H, w1):
	H_w2 = H[:,3:]
	return H_w2.dot(w1)

# Find a point lying at the intersection of 2 polytopes cones for a given intersection, maximizing the distance
# to bounds, assuming dL = 0
# posing H = [H1, H2]^T
# Find x=[c_x, c_y, c_z, s]
# min -s  ([0,0,0,-1]* x)
# K_1 x + k<= 0
# K and k are  computed as follows, given that w1 is the first three lines of w: m * (ddc - g)
# H = [H_w1 ; H_w2]
# H_w1 * c_cross *  w1 + H_w2 * w1 <= 0
# H_w1 * -w1_cross * c + H_w2 * w1 <= 0
# K * c + k <= 0
# then adding s	
# [K, 1_n] *  x + k<= 0
#  K_1 x + k<= 0
#  \param H1 CWC for 1 contact phase
#  \param H2 CWC for the consecutive phase (at least one contact remains between phase, hence intersection exists, but maybe not for the acceleration)
#  \param ddc selected acceleration
#  \param mu friction coefficient
#  \return the point lying at the intersection or an error
def find_intersection_point(H1, H2, ddc, m = 54., g_vec=array([0.,0.,9.81])):
	H = __compute_H(H1, H2);
	w1 = m * (ddc - g_vec);
	K_1 = __compute_K_1(H, w1)
	k = __compute_k(H, w1)
	cost = array([0.,0.,0.,-1.])
	lb =  array([-10000. for _ in range(4)])
	ub =  array([ 10000. for _ in range(4)])
	Alb = array([-10000. for _ in range(k.shape[0])])
	global solver
	(status, res, _) = solver.solve(cost, lb, ub, A_in=K_1, Alb=Alb, Aub=-k, A_eq=None, b=None)
	return status == solver_LP_abstract.LP_status.OPTIMAL, res
