"""
Created on march 6 2017

@author:  stonneau
"""

import sys
sys.path.insert(0, './tools')

from stability import test_eq_cwc
from plot_cond import plot_cond
from transformations import rotation_matrix, identity_matrix
from numpy import array, cross, zeros
from CWC_methods import compute_CWC, is_stable
from lp_dynamic_eq import dynamic_equilibrium_lp
from lp_find_point import find_valid_c_cwc, find_valid_ddc_cwc, find_valid_c_ddc_cwc, find_valid_c_ddc_random

import numpy as np
import math

mu = 0.8
mass = 54.

#reference rectangle contact
p = [np.array([x,y,0,1]) for x in [-0.05,0.05] for y in [-0.1,0.1]]
z = np.array([0,0,1,1])
z_axis = np.array([0.,0.,1.])
y_axis = np.array([0.,1.,0.])
x_axis = np.array([1.,0.,0.])
z = np.array([0,0,1,1])
g = np.array([0.,0.,-9.81])

def gen_contact(center = np.array([0,0,0]),R = identity_matrix()):
	print "R, ", R
	c_4 = np.array(center.tolist()+[0])
	p_rot = [R.dot(p_i + c_4)[0:3] for p_i in p ]
	n_rot = [R.dot(z)[0:3] for _ in range(4) ]
	return np.array(p_rot), np.array(n_rot)
	
P0, N0 = gen_contact(center = np.array([0,0,0]),R = identity_matrix())
P1, N1 = gen_contact(center = np.array([1,0,0]),R = rotation_matrix(math.pi/8., y_axis))
P2, N2 = gen_contact(center = np.array([4,0,0]),R = identity_matrix())

def gen_phase(p_a, n_a, p_b, n_b):
	phase_p = np.zeros((p_a.shape[0] + p_b.shape[0],3))
	phase_n = np.zeros((n_a.shape[0] + n_b.shape[0],3))
	phase_p[0:4,:] = p_a[:];
	phase_n[0:4,:] = n_a[:];
	phase_p[4:8,:] = p_b[:];
	phase_n[4:8,:] = n_b[:];
	return phase_p, phase_n

phase_p_1, phase_n_1 = gen_phase(P0, N0, P1, N1)
phase_p_2, phase_n_2 = P1[:], N1[:]
phase_p_3, phase_n_3 = gen_phase(P1, N1, P2, N2)
#~ phase_p_3, phase_n_3 = P2, N2

H1 = compute_CWC(phase_p_1,phase_n_1,mass=mass,mu=mu)
H2 = compute_CWC(phase_p_2,phase_n_2,mass=mass,mu=mu)
H3 = compute_CWC(phase_p_3,phase_n_3,mass=mass,mu=mu)

#~ assert (is_stable(H1, wp_1[0:3]) and is_stable(H2, wp_1[0:3])), "found com pos is not in both cones H1, H2"

from numpy.random import rand

def _print_res(phasenum, H, c, ddc, phase_p, phase_n, mass, mu):	
	print "phase " + str(phasenum) + " eq (CWC / LP)? "
	res_cwc =  is_stable(H,c,ddc,)
	res_lp, robustness =  dynamic_equilibrium_lp(c, ddc, phase_p, phase_n, mass = mass, mu = mu)
	if(res_cwc != res_lp):
		print "[ERROR] CWC and LP do not agree: (CWC / LP / Robustness )", res_cwc , res_lp, robustness
	print "lp found equiliribum to be : ", res_lp, "with margin ", robustness
	
def test_find_c(H):
	
	ddc= array([ 0.04380291,  0.67393901,  0.7374873 ])
	c= zeros(3)	
	status, sol_found, wp_1 = find_valid_c_cwc(H, ddc, m = mass)
	if(status != 0):
		print "[ERROR] LP find_intersection_c is not feasible"
		return
		#~ 
	c= wp_1[0:3][:]			
	print "solution, (c / ddc) ", c , " " , ddc, "margin", wp_1[3]
	
	_print_res(1, H1, c, ddc, phase_p_1, phase_n_1, mass, mu)	
	_print_res(2, H2, c, ddc, phase_p_2, phase_n_2, mass, mu)	
	_print_res(3, H3, c, ddc, phase_p_3, phase_n_3, mass, mu)	

def test_find_c_ddc(H,mu = 0.6):
	#~ [(c,ddc), success, margin] = find_valid_c_ddc(H, ddc=array([ 0.04380291,  0.67393901,  0.7374873 ]), m = mass)	
	[(c,ddc), success, margin] = find_valid_c_ddc_cwc(H, m = mass)	
	print "solution found ? ", success
	print "Best solution, (c / ddc) ", c , " " , ddc, "margin", margin
	
	_print_res(1, H1, c, ddc, phase_p_1, phase_n_1, mass, mu)	
	_print_res(2, H2, c, ddc, phase_p_2, phase_n_2, mass, mu)	
	_print_res(3, H3, c, ddc, phase_p_3, phase_n_3, mass, mu)	
	
def test_find_c_ddc_rand(H,mu = 0.6):
	#~ [(c,ddc), success, margin] = find_valid_c_ddc(H, ddc=array([ 0.04380291,  0.67393901,  0.7374873 ]), m = mass)	
	[(c,ddc), success, margin] = find_valid_c_ddc_random(H, m = mass)	
	print "solution found ? ", success
	print "Best solution, (c / ddc) ", c , " " , ddc, "margin", margin
	
	_print_res(1, H1, c, ddc, phase_p_1, phase_n_1, mass, mu)	
	_print_res(2, H2, c, ddc, phase_p_2, phase_n_2, mass, mu)	
	_print_res(3, H3, c, ddc, phase_p_3, phase_n_3, mass, mu)	
	
	
#~ print "*********** TEST H1 ********"
#~ test(H1)
print "*********** TEST H2 ********"
test_find_c(H2)
test_find_c_ddc(H2)
test_find_c_ddc_rand(H2)
#~ print "*********** TEST H3 ********"
#~ test(H3)
