"""
Created on march 6, 2017

@author: stonneau
"""

from numpy import array, vstack, zeros, sqrt, cross
import numpy as np

from compute_CWC import compute_CWC

g_vec = array([0,0,-9.81])
                                        
#test stability via polytope projection                                                                                
def test_eq_cwc(P,N,mu,m):	
	H = compute_CWC(P,N,mu=mu)
	def eval_wrench(c,ddc):
		y = m * (ddc - g_vec)
		w = array(y.tolist() + (cross(c, y)).tolist())
		return (H.dot(-w)<=0).all()
	return eval_wrench

#test stability via lp

from pinocchio_inv_dyn.multi_contact.com_acc_LP_3d import ComAccLP3d
from pinocchio_inv_dyn.multi_contact.com_acc_LP import ComAccLP


def _normalize(a, ret_if_zero = np.array([0,0,0])):	
	if(np.linalg.norm(a) != 0):
		return a / np.linalg.norm(a)
	else:
		return ret_if_zero
	

def __max_acc(solver, c, ddc_norm):
	solver.set_com_state(c, -ddc_norm)
	try:
		(imode, max_acc, par_ddc_par_c, act_set_mat, act_set_lb) = solver.compute_max_deceleration(0);
		if(imode!=0):
			#~ print "LP3d failed!";
			return np.array([0,0,0]), False;
		else:
			return max_acc, True
	except ValueError:
			print "lp value error"
			return np.array([0,0,0]), False;

def __in_between(acc_0, acc_1, ddc, norm_ddc):
	alpha_0 = -np.linalg.norm(acc_0) if norm_ddc.dot(_normalize(acc_0)) < 0  else np.linalg.norm(acc_0)
	alpha_1 = -np.linalg.norm(acc_1) if norm_ddc.dot(_normalize(acc_1)) < 0  else np.linalg.norm(acc_1)
	alpha_in = np.linalg.norm(ddc)
	return (alpha_0 <= alpha_in and  alpha_in <= alpha_1) or (alpha_1 <= alpha_in and  alpha_in <= alpha_0)


#minimum com acceleration in direction v (i.e. the maximum acceleration in direction -v).
def _lp(solver,c,ddc):
	norm_ddc = _normalize(ddc, np.array([1,0,0]))
	acc_0, ok = __max_acc(solver, c, -norm_ddc)
	if not ok:
		return False;	
	if np.linalg.norm(ddc == 0):		
		acc_1, ok = __max_acc(solver, c, norm_ddc)	
		if(ok and acc_0 * acc_1 <= 0):	
			print "acc_0, ", acc_0
			print "acc_1, ", acc_1
			print "ddc, ", ddc
		#~ return ok and _normalize(acc_0).dot(_normalize(acc_1)) <= 0
		return ok and acc_0 * acc_1 <= 0
	else:
		return _normalize(acc_0).dot(norm_ddc) > 0 and np.linalg.norm(acc_0) >=  np.linalg.norm(ddc)

#~ def test_eq_lp(P,N,mu,m):	
	#~ def eval_wrench(c,ddc):		
		#~ return _lp(P,N,c,ddc)
	#~ return eval_wrench
	
def test_eq_lp(P,N,mu,m):	
	solver = ComAccLP("pff", c0=np.array([0,0,0]), v=np.array([1,0,0]), contact_points=P, contact_normals=N, mu=0.3, g = g_vec, mass=54, 
                  maxIter=10000, verb=0, regularization=1e-5)
	solver.NO_WARM_START = True
	def eval_wrench(c,ddc):		
		return _lp(solver,c,ddc)
	return eval_wrench
	
