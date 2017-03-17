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
from lp_dynamic_eq import dynamic_equilibrium

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
increments = [0.05,0.05,0.2]

#~ bounds = [-.5,.5,-.5,.5,-0,2.1]
#~ increments = [0.1,0.1,1]

#~ P,N = gen_contact()
def test(beta = 0, mu = 0.3, ddc = np.array([0,0,0]), mass = 54., method = test_eq_lp):
	P,N = gen_contact(R = rotation_matrix(beta, y_axis))
	return plot_cond(P,N,bounds,increments, ddc, method(P,N,mu,mass))

import matplotlib.pyplot as plt


def seq_test(beta=0., ddc=np.array([1,0,0])):
	test(beta,ddc=np.array([1,0,0]), method = test_eq_lp)
	test(beta,ddc=np.array([1,0,0]), method = test_eq_cwc)
	plt.show()
	
seq_test(-0.,np.array([1,0,1]))

from numpy.random import uniform
from math import pi

def _rand_array(size=3, lb=-5., ub=5.):
	return np.array([uniform(lb,ub) for _ in range(3)])

def __gen_values(min_point, max_point, inc):
	assert(max_point > min_point)
	num_points = int(float(max_point - min_point) / float(inc))
	return [min_point + inc * v for v in range(num_points)]

def __gen_points_in_bounds(bounds, discretizationSteps):	
	x_vals = __gen_values(bounds[0],bounds[1],discretizationSteps[0])
	y_vals = __gen_values(bounds[2],bounds[3],discretizationSteps[1])
	z_vals = __gen_values(bounds[4],bounds[5],discretizationSteps[2])
	return  [np.array([x_i,y_i,z_i]) for  x_i in x_vals for  y_i in y_vals for  z_i in z_vals]

#randomly sample angles and acceleration, and count, for the same acceleration,
# the differences in equilibrium successes
def compare_cones(nb_iter_ddc=100, nb_iter_angle=2, angles = None):
	nb_succ = 0.;
	nb_fail = 0.;
	ddcs = [_rand_array() for _ in  range(nb_iter_ddc)]
	if(angles == None):
		angles = [uniform(-pi/2.,pi/2.) for _ in  range(nb_iter_angle)]
	c_s = __gen_points_in_bounds(bounds, increments)
	ddcs_cs = [(ddc, c) for ddc in ddcs for c in c_s]
	
	for (ddc, c) in ddcs_cs:
		concordant = True
		valid = None
		for angle in angles:
			P,N = gen_contact(R = rotation_matrix(angle, y_axis))
			stable = dynamic_equilibrium(c, ddc, P, N, mass = 54., mu = 0.3)
			if(stable):
				print 'stable, ', angle
			if(valid != None and valid != stable):
				concordant = False
			valid = stable
		if(concordant and valid):
			nb_succ += 1
		elif(not concordant):
			nb_fail +=1
	print 'ddcs:', ddcs
	print 'angles:', angles
	print 'successes:', nb_succ
	print 'failures :', nb_fail
	print 'success ratio :', nb_succ / (nb_succ + nb_fail)
	
	
a = compare_cones(1,2, [0, 0.4])

