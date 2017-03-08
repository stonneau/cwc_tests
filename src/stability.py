"""
Created on march 6, 2017

@author: stonneau
"""

from numpy import array, vstack, zeros, sqrt, cross
import numpy as np

from compute_CWC import compute_CWC
from lp_dynamic_eq import dynamic_equilibrium

g_vec = array([0,0,-9.81])
                                        
#test stability via polytope projection                                                                                
def test_eq_cwc(P,N,mu,m):	
	H = compute_CWC(P,N,mu=mu)
	def eval_wrench(c,ddc):
		y = m * (ddc - g_vec)
		w = array(y.tolist() + (cross(c, y)).tolist())
		return (H.dot(w)<=0).all()
	return eval_wrench

	
def test_eq_lp(P,N,mu,m):	
	def eval_wrench(c,ddc):		
		return dynamic_equilibrium(c, ddc, P, N, m, mu)
	return eval_wrench



	
