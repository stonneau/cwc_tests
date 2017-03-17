"""
Created on Tue March 7

@author: stonneau
"""

import os, sys
sys.path.insert(0, './tools')

from polytope_conversion_utils import crossMatrix
from numpy import array, zeros, ones, sqrt, cross, identity
NUMBER_TYPE = 'float'  # 'float' or 'fraction'
__EPS =  2.39847622652e-4
#~ g_vec = array([0,0,-9.81])

#~ from scipy.optimize import linprog
from pinocchio_inv_dyn.optimization import solver_LP_abstract
 

def __compute_K_1(K):
	K_1 = ones((K.shape[0],4))
	K_1[:,:3] = K[:]
	return K_1
 
 
def lp_ineq_4D(K,k):
	K_1 = __compute_K_1(K)
	cost = array([0.,0.,0.,-1.])
	lb =  array([-10000. for _ in range(4)]); lb[2]=0.;
	ub =  array([ 10000. for _ in range(4)])
	#~ K_1 = K
	#~ cost = ones(3)
	#~ lb =  array([-10000. for _ in range(3)]); #lb[3]=0.;
	#~ ub =  array([ 10000. for _ in range(3)])
	Alb = array([-10000. for _ in range(k.shape[0])])
	solver = solver_LP_abstract.getNewSolver('qpoases', "dyn_eq", maxIter=10000, maxTime=100.0, useWarmStart=False, verb=1)
	(status, res, _) = solver.solve(cost, lb = lb, ub = ub, A_in=K_1, Alb=Alb, Aub=-k, A_eq=None, b=None)
	
	print "eq satisfied ?", res[3]
	
	print (K_1.dot(res) + k <= __EPS).all(), K_1.shape, k.shape, res.shape, (K_1.dot(res) + k).shape, max((K_1.dot(res) + k).T )
	c = zeros(3); c[:]=  res[0:3]
	print (K.dot(res[0:3]) + k <= __EPS).all(), K.shape, k.shape, res[0:3].shape, (K.dot(c) + k).shape, max((K.dot(c) + k).T )
	
	#problem solved or unfeasible
	p_solved = solver_LP_abstract.LP_status.OPTIMAL and res[3] >= 0.
	status_ok = status== solver_LP_abstract.LP_status.OPTIMAL
	return status, status_ok and res[3] >= 0., res
	#~ return status, status_ok , res
	
 
#********* BEGIN find_intersection_c ********************
def __compute_H(H1, H2):
	print "H1.shape", H1.shape
	print "H2.shape", H2.shape
	assert H1.shape[1] == H2.shape[1], "matrix do not have the same dimension"
	H_tot = zeros((H1.shape[0]+ H2.shape[0],H1.shape[1]));
	H_tot[0:H1.shape[0],:] = H1[:]
	H_tot[H1.shape[0]:,:] = H2[:]
	return H_tot
	#~ return H1
	
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
# posing H = [H1, H2]^T
# Find x=[c_x, c_y, c_z, s]
# min -s  ([0,0,0,-1]* x)
# K_1_c x + k_c<= 0
# K_1_c and k_c are  computed as follows, given that w1 is the first three lines of w: m * (ddc - g)
# H = [H_w1 ; H_w2]
# H_w2 * c_cross *  w1 + H_w1 * w1 <= 0
# H_w2 * -w1_cross * c + H_w1 * w1 <= 0
# K_c * c + k_c <= 0
# then adding s	
# [K_c, 1_n] *  x + k_c<= 0
#  K_1_c x + k_c<= 0
#  \param H1 CWC for 1 contact phase
#  \param H2 CWC for the consecutive phase (at least one contact remains between phase, hence intersection exists, but maybe not for the acceleration)
#  \param ddc selected acceleration
#  \param mu friction coefficient
#  \return the solver status, whether the point satisfies the constraints, and the closest point that satisfies them
def find_valid_c(H, ddc, m = 54., g_vec=array([0.,0.,-9.81])):
	w1 = m * (ddc - g_vec)
	K_c = __compute_K_c(H, w1)
	k_c = __compute_k_c(H, w1)
	return lp_ineq_4D(K_c,k_c)
	#~ cost = array([0.,0.,0.,-1.])
	#~ lb =  array([-10000. for _ in range(4)]); #lb[3]=0.;
	#~ ub =  array([ 10000. for _ in range(4)])
	#~ Alb = array([-10000. for _ in range(k_c.shape[0])])
	#~ solver = solver_LP_abstract.getNewSolver('qpoases', "dyn_eq", maxIter=1000, maxTime=100.0, useWarmStart=True, verb=3)
	#~ (status, res, _) = solver.solve(cost, lb = lb, ub = ub, A_in=K_1, Alb=Alb, Aub=-k_c, A_eq=None, b=None)
	#~ 
	#~ #problem solved or unfeasible
	#~ p_solved = solver_LP_abstract.LP_status.OPTIMAL and res[3] >= 0
	#~ status_ok = status== solver_LP_abstract.LP_status.OPTIMAL
	#~ return status, status_ok and res[3] >= 0, res

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
#  \param H1 CWC for 1 contact phase
#  \param H2 CWC for the consecutive phase (at least one contact remains between phase, hence intersection exists, but maybe not for the acceleration)
#  \param ddc selected acceleration
#  \param mu friction coefficient
#  \return the solver status, whether the point satisfies the constraints, and the closest point that satisfies them
def find_valid_ddc(H, c, m = 54., g_vec=array([0.,0.,-9.81])):
	K_ddc = __compute_K_ddc(m, H, c)
	k_ddc   = __compute_k_ddc  (m, H, c, g_vec)
	return lp_ineq_4D(K_ddc,k_ddc)
	

#********* END find_intersection_ddc ********************
