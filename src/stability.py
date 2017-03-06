"""
Created on Thurs Aug 4

@author: adelpret, updated by stonneau
"""

from numpy import array, vstack, zeros, sqrt, cross
import numpy as np

from compute_CWC import compute_CWC

g_vec = array([0,0,-9.81])
                                        
#test stability via polytope projection                                                                                
def test_eq_cwc(P,N,mu):	
	H = compute_CWC(P,N,mu=mu)
	def eval_wrench(c,ddc,m):
		y = m * (ddc - g_vec)
		w = array(y.tolist() + (cross(c, y)).tolist())
		return (H.dot(-w)<=0).all()
	return eval_wrench

#test stability via lp
