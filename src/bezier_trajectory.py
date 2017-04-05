"""
Created on march 6 2017

@author:  stonneau
"""

import sys
sys.path.insert(0, './tools')

from transformations import rotation_matrix, identity_matrix
from numpy import array, cross, zeros, matrix, asmatrix, asarray
import numpy as np
import math
from lp_dynamic_eq import dynamic_equilibrium_lp
from lp_find_point import find_valid_c_cwc, find_valid_ddc_cwc, find_valid_c_ddc_cwc, find_valid_c_ddc_random

from CWC_methods import compute_w, compute_CWC, is_stable

#importing bezier routines
from spline import bezier6 as bezier

from numpy.random import rand

#compute integral such that init point is c_0
def __compute_c_t(b_curve, c_0, c_end, g_vec):
	wc_t = b_curve.compute_primitive(2)
	delta_ = c_0 - wc_t(0.)[0:3,:] #make sure c_0 is reached
	c_end_offset = c_end - g_vec / 2. - delta_
	wps = wc_t.waypoints()
	wps[0:3,-1] = c_end_offset
	wc_t = bezier(wps)
	def c_of_t(t):
		print "wc_t, " ,wc_t(0.)[0:3,:]
		print "c_0, " ,c_0
		print "delta, " ,delta_
		print "wc_t(t), " ,wc_t(t)[0:3,:]
		print "g_vec, " ,g_vec
		return asarray(delta_ + wc_t(t)[0:3,:] + 0.5 * g_vec * t * t).flatten()
	return c_of_t, wc_t

def __compute_ddc_t(b_curve, m, g_vec):	
	def ddc_of_t(t):
		print "b_curve(t) ", b_curve(t)
		return asarray((b_curve(t)[0:3,:] / m) + g_vec).flatten()
	return ddc_of_t
	
	
def __compute_c_ddc_t(w1_t, c_0, c_end, m, g_vec):	
	__c = asmatrix(c_0).T
	__c_end = asmatrix(c_end).T
	__g = asmatrix(g_vec).T
	c_of_t, c_of_t_bez   = __compute_c_t(w1_t, __c, __c_end, __g)
	#~ ddc_of_t = __compute_ddc_t(w1_t, m, __g)
	ddc_of_t = c_of_t_bez.compute_derivate(2)
	def c_ddc_of_t(t):
		return (c_of_t(t), asarray(ddc_of_t(t)).flatten()[3:6])
	return c_ddc_of_t
	
# Computes a com trajectory between points lying in the same cone
# the trajectory returned is (c,ddc)(t). It passes exactly at the start and end configurations,
# and is guaranteed to lie in the cone.
# it is computed as follows:
#	a) compute bezier curve between two wrenches w_start and w_end, w(t)
#   b) assuming dL = 0, extract w1(t) =  m ddc(t) - mg. This gives ddc(t) = (w1(t) - mg) / m
#   c) integrate ddc(t) twice to obtain c(t): c(t) = [w1(t)]^2 + g * t*t / 2
#   d) modify c(t) end waypoint to reach desired end position s.t. p_end =  c_end - g / 2
#   e) derivate twice again to compute acceleration
#  \param cs_ddcs list of doubles (c,_ddc) indicating the waypoints. Must be at least of size 2
#  \param m mass of the robot
#  \param g_vec gravity acceleration
#  \return (c,ddc)(t)
def bezier_traj(cs_ddcs, m = 54., g_vec=array([0.,0.,-9.81]), mu = 0.6):
	assert len(cs_ddcs) >= 2, "cannot create com trajectory with less than 2 points"
	#creating waypoints for curve
	
	waypoints = matrix([compute_w(c, ddc, m = m, g_vec = g_vec)	for (c,ddc) in cs_ddcs]).transpose()
	print "waypoints", waypoints
	w1_t = bezier(waypoints)
	#~ return w1_t
	#~ return __compute_c_t(w1_t, cs_ddcs[0][0], g_vec)
	return __compute_c_ddc_t(w1_t, cs_ddcs[0][0], cs_ddcs[-1][0], m, g_vec)
	return w1_t
	

def eval_valid_part(P, N, traj, step = 0.1, m = 54., g_vec=array([0.,0.,-9.81]), mu = 0.6):
	num_steps = int(1./step)
	previous_c_ddc = traj(0)
	for i in range(1, num_steps):
		(c,ddc) = traj(float(i)*step)
		res_lp, robustness =  dynamic_equilibrium_lp(c, ddc, phase_p_1, phase_n_1, mass = m, mu = mu)
		if(robustness >= 0.):
			previous_c_ddc = (c,ddc)
		else:
			return False, compute_w(previous_c_ddc[0],previous_c_ddc[1], m = m, g_vec = g_vec), float(i-1) * step
	return True, compute_w(previous_c_ddc[0],previous_c_ddc[1], m = m, g_vec = g_vec), 1.

if __name__ == '__main__':
	
	#importing test contacts
	from contact_data_test import *
	g_vec=array([0.,0.,-9.81])
	[c0_ddc0, success, margin] = find_valid_c_ddc_random(phase_p_1, phase_n_1, m = mass, mu = mu)
	[c1_ddc1, success, margin] = find_valid_c_ddc_random(phase_p_1, phase_n_1, m = mass, mu = mu)
	b = bezier_traj([c0_ddc0, c1_ddc1])
	res = eval_valid_part(phase_p_1, phase_n_1, b, step = 0.1, m = mass, g_vec=g_vec, mu = mu)
	
	#find two points in the cone
	
	#~ m = 54.
	#~ g_vec=array([0.,0.,-9.81])
	#~ c0_ddc0 = (array([10. for _ in range(3)]), array([1. for _ in range(3)]))
	#~ c1_ddc1 = (array([25. for _ in range(3)]), array([3. for _ in range(3)]))
	#~ b = bezier_traj([c0_ddc0, c1_ddc1])
