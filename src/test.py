"""
Created on march 6 2017

@author:  stonneau
"""

import sys
sys.path.insert(0, './tools')

from compute_CWC import compute_CWC
from plot_static_equilibrium import plot_quasi_static_feasible_c
from transformations import rotation_matrix, identity_matrix
from numpy import array, cross

import numpy as np

# params
mass = 1


#reference rectangle contact
p = [np.array([x,y,0,1]) for x in [-0.05,0.05] for y in [-0.1,0.1]]
z = np.array([0,0,1,1])
g = np.array([0,0,-9.81])

def gen_contact(center = np.array([0,0,0]),R = identity_matrix()):
	c_4 = np.array(center.tolist()+[0])
	p_rot = [R.dot(p_i + c_4)[0:3] for p_i in p ]
	n_rot = [R.dot(z)[0:3] for _ in range(4) ]
	return np.array(p_rot), np.array(n_rot)
	
	
P,N = gen_contact()
H = compute_CWC(P,N)
	

plot_quasi_static_feasible_c(H,mass,[-1,1,-1,1,0,2],[0.02,0.02,0.2], P)

	


