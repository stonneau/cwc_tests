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
from compute_CWC import compute_CWC, is_stable
from lp_dynamic_eq import dynamic_equilibrium
from lp_intersect import find_intersection_point

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
	
P0, N0 = gen_contact(center = np.array([0,0,0]),R = identity_matrix())
P1, N1 = gen_contact(center = np.array([1,0,0]),R = rotation_matrix(0, y_axis))
P2, N2 = gen_contact(center = np.array([2,0,0]),R = rotation_matrix(-50, y_axis))

def gen_phase(p_a, n_a, p_b, n_b):
	phase_p = np.zeros((p_a.shape[0] + p_b.shape[0],3))
	phase_n = np.zeros((n_a.shape[0] + n_b.shape[0],3))
	phase_p[0:4,:] = p_a;
	phase_n[0:4,:] = n_a;
	phase_p[4:8,:] = p_b;
	phase_n[4:8,:] = n_b;
	return phase_p, phase_n

phase_p_1, phase_n_1 = gen_phase(P0, N0, P1, N1)
phase_p_2, phase_n_2 = P1, N1
phase_p_3, phase_n_3 = gen_phase(P1, N1, P2, N2)

H1 = compute_CWC(phase_p_1,phase_n_1,mu=0.3)
H2 = compute_CWC(phase_p_2,phase_n_2,mu=0.3)
H3 = compute_CWC(phase_p_3,phase_n_3,mu=0.3)

res, wp_1 = find_intersection_point(H1,H2, np.array([0.,0.,0.]))

print is_stable(H1, wp_1[0:3])
print is_stable(H2, wp_1[0:3])

assert (is_stable(H1, wp_1[0:3]) and is_stable(H2, wp_1[0:3])), "found com pos is not in both cones H1, H2"
