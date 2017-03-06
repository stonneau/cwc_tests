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

	
	#minimum com acceleration in direction v (i.e. the maximum acceleration in direction -v).
def test_eq_lp(P,N,mu,m):	
	H = compute_CWC(P,N,mu=mu)
	solver = ComAccLP3d("pff", c0=np.array([0,0,0]), v=np.array([1,0,0]), contact_points=P, contact_normals=N, mu=mu, g = g_vec, mass=m, 
                  maxIter=10000, verb=0, grad_reg=1e-5)
	def eval_wrench(c,ddc):
		norm_acc = ddc / np.linalg.norm(ddc)
		#minimum com acceleration in direction v (i.e. the maximum acceleration in direction -v).
		solver.set_com_state(c, -norm_acc);
		print "ddc", ddc
		imode, max_acc, par_ddc_par_c, act_set_mat, act_set_lb = solver.compute_max_deceleration();
		print "ddc2", ddc
        return ddc.dot(max_acc) > 0 and (np.linalg.norm(ddc) < np.linalg.norm(max_acc));
        #~ if(imode!=0):
            #~ print "LP3d failed!";
	return eval_wrench
