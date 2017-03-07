"""
Created on march 6 2017

@author:  stonneau
"""

import sys
sys.path.insert(0, './tools')

from stability import test_eq_cwc
from plot_cond import plot_cond
from transformations import rotation_matrix, identity_matrix
from numpy import array, cross
from stability import test_eq_cwc, test_eq_lp

import numpy as np


#reference rectangle contact
p = [np.array([x,y,0,1]) for x in [-0.05,0.05] for y in [-0.1,0.1]]
z = np.array([0,0,1,1])
z_axis = np.array([0,0,1])
y_axis = np.array([0,1,0])
x_axis = np.array([1,0,0])
z = np.array([0,0,1,1])
g = np.array([0,0,-9.81])

def gen_contact(center = np.array([0,0,0]),R = identity_matrix()):
	c_4 = np.array(center.tolist()+[0])
	p_rot = [R.dot(p_i + c_4)[0:3] for p_i in p ]
	n_rot = [R.dot(z)[0:3] for _ in range(4) ]
	return np.array(p_rot), np.array(n_rot)
	

bounds = [-1,1,-1,1,-0,2.1]
increments = [0.01,0.01,1]

bounds = [-.5,.5,-.5,.5,-0,2.1]
increments = [0.1,0.1,1]

#~ P,N = gen_contact()
def test(beta = 0, mu = 0.3, ddc = np.array([0,0,0]), mass = 54, method = test_eq_lp):
	P,N = gen_contact(R = rotation_matrix(beta, y_axis))
	print "R", rotation_matrix(beta, y_axis)
	print "P", P
	print "N", N
	global H
	plot_cond(P,N,bounds,increments, ddc, method(P,N,mu,mass))
	#~ plot_cond(P,N,bounds,increments, ddc, test_wtf)


test(0.,ddc=np.array([0,0,0]), method = test_eq_lp)
#~ test(0.,ddc=np.array([0,0,0]))


