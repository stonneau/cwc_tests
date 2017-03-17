"""
Created on march 6, 2017

@author: stonneau
"""

from numpy import array, vstack, zeros, sqrt, cross, asmatrix
import numpy as np

from lp_dynamic_eq import dynamic_equilibrium_lp
from CWC_methods import compute_w, compute_CWC, is_stable

                         
from centroidal_dynamics import *
               
#test stability via polytope projection                                                                                
def test_eq_cwc(P,N,mu,m, g_vec = array([0,0,-9.81])):	
	H = compute_CWC(P,N,mu=mu)	
	def eval_wrench(c,ddc, dL=array([0.,0.,0.])):
		return is_stable(H, c, ddc=ddc, dL=dL, m=m, g_vec=g_vec)
	return eval_wrench

	
def test_eq_lp(P,N,mu,m):	
	def eval_wrench(c,ddc):		
		return dynamic_equilibrium_lp(c, ddc, P, N, m, mu)[0]
	return eval_wrench



	
