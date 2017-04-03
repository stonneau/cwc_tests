"""
Created on Tue March 7

@author: stonneau
"""

import os, sys
sys.path.insert(0, './tools')

from polytope_conversion_utils import crossMatrix
from numpy import array, zeros, ones, sqrt, cross, identity, asmatrix
NUMBER_TYPE = 'float'  # 'float' or 'fraction'
__EPS =  2.39847622652e-4
__ACC_MARGIN =  0
#~ __EPS =  0

#~ from scipy.optimize import linprog
from pinocchio_inv_dyn.optimization import solver_LP_abstract

def __compute_K_1(K):
	K_1 = ones((K.shape[0],4))
	K_1[:,:3] = K[:]
	return K_1
 
 
def lp_ineq_4D(K,k):
	K_1 = __compute_K_1(K)
	cost = array([0.,0.,0.,-1.])
	lb =  array([-100000000. for _ in range(4)]);
	ub =  array([ 100000000. for _ in range(4)])
	Alb = array([-100000000. for _ in range(k.shape[0])])
	solver = solver_LP_abstract.getNewSolver('qpoases', "dyn_eq", maxIter=10000, maxTime=10000.0, useWarmStart=True, verb=0)
	(status, res, rest) = solver.solve(cost, lb = lb, ub = ub, A_in=K_1, Alb=Alb, Aub=-k, A_eq=None, b=None)
		
	#problem solved or unfeasible
	p_solved = solver_LP_abstract.LP_status.OPTIMAL and res[3] >= 0.
	status_ok = status== solver_LP_abstract.LP_status.OPTIMAL
	return status, status_ok , res
	
 
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
# then adding s	
# [K_c, 1_n] *  x + k_c<= 0
#  K_1_c x + k_c<= 0
#  \param H CWC for a contact phase
#  \param ddc selected acceleration
#  \param mu friction coefficient
#  \param g_vec gravity acceleration
#  \return the solver status, whether the point satisfies the constraints, and the closest point that satisfies them
def find_valid_c(H, ddc, m = 54., g_vec=array([0.,0.,-9.81])):
	w1 = m * (ddc - g_vec)
	K_c = __compute_K_c(H, w1)
	k_c = __compute_k_c(H, w1)
	return lp_ineq_4D(K_c,k_c)

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
def find_valid_ddc(H, c, m = 54., g_vec=array([0.,0.,-9.81])):
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
def find_valid_c_ddc(H, max_iter = 5, ddc=array([0.,0.,0.]), m = 54.,  g_vec=array([0.,0.,-9.81])):
	current_iter = max_iter
	#~ __c = c[:]
	__ddc = ddc[:]
	
	#~ while(current_iter > 0):
		#~ print "ITER "
		#~ current_iter -= 1
		#~ status, sol_found, wp_1 = find_valid_ddc(H, __c)
		#~ if(status != 0):
				#~ print "[ERROR] LP find_valid_ddc is not feasible"
				#~ return
		#~ ddc = wp_1[0:3]
		#~ status, sol_found, wp_1 = find_valid_c(H, ddc, m = m, g_vec = g_vec)
		#~ if(status != 0):
				#~ print "[ERROR] LP find_valid_c is not feasible"
				#~ return
		#~ __c =  wp_1[0:3][:]
		#~ margin = wp_1[3]	
		#~ sol_found = margin >=0.1
		 #~ 
		#~ if sol_found:
			#~ print "solution, (c / ddc) ", __c , " " , ddc, "margin", margin
			#~ print "number of iterations required ", max_iter - current_iter
			#~ break;
		#~ 
	#~ 
	#~ return [(__c,ddc), sol_found, margin]	 
	
	while(current_iter > 0):
		current_iter -= 1
		status, sol_found, wp_1 = find_valid_c(H, __ddc, m = m, g_vec = g_vec)
		if(status != 0):
			print "[ERROR] LP find_intersection_c is not feasible"
			return
		c= wp_1[0:3][:]
		
		margin = wp_1[3]	
		sol_found = sol_found and margin > __ACC_MARGIN	 
		
		if(sol_found):
			print "FOUND DIRECTLY solution, (c / ddc) ", c , " " , __ddc, "margin", margin
			return [(c,__ddc), True, margin]
		#~ if(not sol_found):
		if(True):
			print "no solution found for acceleration " , __ddc , " (margin ", margin, ") , try to find acceleration with best c", c
			status, sol_found, wp_1 = find_valid_ddc(H, c)
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
