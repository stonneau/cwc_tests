"""
Created on march 6 2017

@author:  stonneau
"""

import sys
sys.path.insert(0, './tools')

from transformations import rotation_matrix, identity_matrix
from numpy import array, cross, zeros, matrix, asmatrix, asarray
from numpy.linalg import norm
import numpy as np
import math
from lp_dynamic_eq import dynamic_equilibrium_lp
from lp_find_point import find_valid_c_cwc, find_valid_ddc_cwc, find_valid_c_ddc_cwc, find_valid_c_ddc_random

from CWC_methods import compute_w, compute_CWC, is_stable

#importing bezier routines
from spline import bezier, curve_constraints

from numpy.random import rand

zero3 = array([0.,0.,0.])

def __compute_ddc_t(b_curve, m, g_vec):	
	def ddc_of_t(t):
		#~ print "b_curve(t) ", b_curve(t)
		return asarray((b_curve(t)[0:3,:] / m) + g_vec).flatten()
	return ddc_of_t
	
	
def __compute_c_ddc_t(c_t):	
	ddc_t = c_t.compute_derivate(2)
	def c_ddc_of_t(t):
		return (asarray(c_t(t)).flatten(), asarray(ddc_t(t)).flatten())
	return c_ddc_of_t
	
# Computes a com trajectory between points lying in the same cone
# the trajectory returned is (c,ddc)(t). It passes exactly at the start and end configurations, and respects
# given constraints on vel/acceleration at start and end phases.
#  \param cs_ddcs list of doubles (c,_ddc) indicating the waypoints. Must be at least of size 2
#  \param init_dc_ddc specified init speed and acceleration
#  \param end_dc_ddc specified end speed and acceleration
#  \return (c,ddc)(t)
def bezier_traj(cs_ddcs, init_dc_ddc = (zero3,zero3), end_dc_ddc = (zero3,zero3)):
	assert len(cs_ddcs) >= 2, "cannot create com trajectory with less than 2 points"
	#creating waypoints for curve	
	waypoints = matrix([ c	for (c,_) in cs_ddcs]).transpose()
	c = curve_constraints();
	c.init_vel = matrix(init_dc_ddc[0]);
	c.end_vel  = matrix(end_dc_ddc[0]);
	c.init_acc = matrix(init_dc_ddc[1]);
	c.end_acc  = matrix(end_dc_ddc[1]);
	
	c_t = bezier(waypoints, c)
	return __compute_c_ddc_t(c_t)
	

def eval_valid_part(P, N, traj, step = 0.1, m = 54., g_vec=array([0.,0.,-9.81]), mu = 0.6, use_cone_for_eq = None, rob = 0):
	num_steps = int(1./step)
	previous_c_ddc = traj(0)
	for i in range(1, num_steps):
		(c,ddc) = traj(float(i)*step)
		if(use_cone_for_eq != None):
			if is_stable(use_cone_for_eq,c=c, ddc=ddc, dL=array([0.,0.,0.]), m = m, g_vec=array([0.,0.,-9.81]), robustness = 0.):
				robustness = 1
			else:
				robustness = -1
		else:
			res_lp, robustness =  dynamic_equilibrium_lp(c, ddc, P, N, mass = m, mu = mu)
		if(robustness >= rob):
			previous_c_ddc = (c,ddc)
		else:
			#~ print "failed with robustness ", robustness, "at ", i, "point ", previous_c_ddc, "rob ?", rob
			return False, previous_c_ddc , float(i-1) * step
	return True, previous_c_ddc, 1.

import matplotlib
#~ matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

g_vec=array([0.,0.,-9.81])

def filter_wps(wps):
	res = []
	#~ res.append(wps[0])
	#~ wps_tmp = reversed(wps[1,-1])
	wps_tmp = wps[1:-1][:]
	assert(len(wps_tmp) == len(wps)-2)
	for i in range(len(wps_tmp)-1):
		if not (norm(wps_tmp[i+1][0]-wps_tmp[i][0]) < 0.01):
			res.append(wps_tmp[i])
		else:
			print "removing, ", wps_tmp[i][0], " (", wps_tmp[i+1][0] ,")"
	res.append(wps_tmp[-1])
	#~ print "len before / after ", len(wps), len(res) +2
	return [wps[0]] + res + [wps[-1]]

def connect_two_points(c0_ddc0, c1_ddc1, P, N, m = 54., mu = 0.3, g_vec = g_vec, use_cone_for_eq = None, plot = False, dt = 0.01, rob = 0):
	#assert positions given are valid
	#~ res_lp, robustness =  dynamic_equilibrium_lp(c0_ddc0[0], c0_ddc0[1], P, N, mass = m, mu = mu)
	#~ assert robustness >= 0., "init config is not in equilibrium (robustness, mu) " + str (robustness) + str(" ") + str (m)
	#~ res_lp, robustness =  dynamic_equilibrium_lp(c1_ddc1[0], c1_ddc1[1], P, N, mass = m, mu = mu)
	#~ assert robustness >= 0., "end config is not in equilibrium(robustness, mu) " + str (robustness) + str(" ") + str (m)
	
	b = bezier_traj([c0_ddc0, c1_ddc1], init_dc_ddc = (zero3,c0_ddc0[1]), end_dc_ddc = (zero3,c1_ddc1[1]))	
	init_traj_ok, c_ddc, step = eval_valid_part(P, N, b, step = 0.1, m = m, g_vec=g_vec, mu = mu)
	
	wps = [c0_ddc0, c1_ddc1]
	init_dc_ddc = (zero3,c0_ddc0[1]);
	end_dc_ddc = (zero3,c1_ddc1[1])
	
	#~ if plot:
		#~ print "plot init trajectory"
		#~ fig = plt.figure()
		#~ ax = fig.add_subplot(111, projection='3d')
		#~ n = 100
		#~ points = [b(0.01 * i)[0] for i in range(100)]
		#~ xs = [point[0] for point in points]
		#~ ys = [point[1] for point in points]
		#~ zs = [point[2] for point in points]
		#~ ax.scatter(xs, ys, zs, c='b')
#~ 
		#~ colors = ["r", "b", "g"]
		#print contact points of first phase
		#~ xs = [point[0] for point in P]
		#~ ys = [point[1] for point in P]
		#~ zs = [point[2] for point in P]
		#~ ax.scatter(xs, ys, zs, c=colors[0])
			#~ 
		#~ ax.set_xlabel('X Label')
		#~ ax.set_ylabel('Y Label')
		#~ ax.set_zlabel('Z Label')
		#~ 
		#~ xs = [point[0] for (point,_) in wps]
		#~ ys = [point[1] for (point,_) in wps]
		#~ zs = [point[2] for (point,_) in wps]
		#~ ax.scatter(xs, ys, zs, c=colors[2])
	
	
	wps = [c0_ddc0, c1_ddc1]
	init_dc_ddc = (zero3,c0_ddc0[1]);
	end_dc_ddc = (zero3,c1_ddc1[1])
	found = init_traj_ok; max_iters = 22;
	while (not (found or max_iters == 0)):
		#~ print "maxtiters", max_iters
		b = bezier_traj(wps, init_dc_ddc = init_dc_ddc, end_dc_ddc = end_dc_ddc)
		found, c_ddc, step = eval_valid_part(P, N, b, step = dt, m = m, g_vec=g_vec, mu = mu, use_cone_for_eq = use_cone_for_eq, rob = rob)
		#~ print "last step valiud at phase: ", step
		if(step == 0.0):
			break
		if(not found):
			wps = wps[:-1] + [c_ddc] + [wps[-1]]
			#~ wps = filter_wps(wps)
		max_iters = max_iters-1;

	
	
	#~ print "found? ", found
	
	#~ if found and plot:
		#~ 
		#~ print "plot trajectory"
		#~ fig = plt.figure()
		#~ ax = fig.add_subplot(111, projection='3d')
		#~ n = 100
		#~ points = [b(0.01 * i)[0] for i in range(100)]
		#~ xs = [point[0] for point in points]
		#~ ys = [point[1] for point in points]
		#~ zs = [point[2] for point in points]
		#~ ax.scatter(xs, ys, zs, c='b')
#~ 
		#~ colors = ["r", "b", "g"]
		#print contact points of first phase
		#~ xs = [point[0] for point in P]
		#~ ys = [point[1] for point in P]
		#~ zs = [point[2] for point in P]
		#~ ax.scatter(xs, ys, zs, c=colors[0])
			#~ 
		#~ ax.set_xlabel('X Label')
		#~ ax.set_ylabel('Y Label')
		#~ ax.set_zlabel('Z Label')
		#~ 
		#print control points
		#~ xs = [point[0] for (point,_) in wps]
		#~ ys = [point[1] for (point,_) in wps]
		#~ zs = [point[2] for (point,_) in wps]
		#~ ax.scatter(xs, ys, zs, c=colors[2])
			#~ 
		#now draw control points
			
		#~ plt.show()
		
	return found, init_traj_ok

if __name__ == '__main__':
	
	
	
	#importing test contacts
	from contact_data_test import *
	g_vec=array([0.,0.,-9.81])
	[c0_ddc0, success, margin] = find_valid_c_ddc_random(phase_p_3, phase_n_3, m = mass, mu = mu)
	[c1_ddc1, success, margin] = find_valid_c_ddc_random(phase_p_2, phase_n_2, m = mass, mu = mu)
	b = bezier_traj([c0_ddc0, c1_ddc1], init_dc_ddc = (zero3,c0_ddc0[1]), end_dc_ddc = (zero3,c1_ddc1[1]))
	
	print "plot init trajectory"
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	n = 100
	points = [b(0.01 * i)[0] for i in range(100)]
	xs = [point[0] for point in points]
	ys = [point[1] for point in points]
	zs = [point[2] for point in points]
	ax.scatter(xs, ys, zs, c='b')

	colors = ["r", "b", "g"]
	#~ #print contact points of first phase
	xs = [point[0] for point in phase_p_3]
	ys = [point[1] for point in phase_p_3]
	zs = [point[2] for point in phase_p_3]
	ax.scatter(xs, ys, zs, c=colors[0])
		
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
		
	#~ plt.show()
	
	wps = [c0_ddc0, c1_ddc1]
	init_dc_ddc = (zero3,c0_ddc0[1]);
	end_dc_ddc = (zero3,c1_ddc1[1])
	
	
	xs = [point[0] for (point,_) in wps]
	ys = [point[1] for (point,_) in wps]
	zs = [point[2] for (point,_) in wps]
	ax.scatter(xs, ys, zs, c=colors[2])
	
	found = False; max_iters = 100;
	while (not (found or max_iters == 0)):
		max_iters = max_iters-1;
		print "maxtiters", max_iters
		b = bezier_traj(wps, init_dc_ddc = init_dc_ddc, end_dc_ddc = end_dc_ddc)
		found, c_ddc, step = eval_valid_part(phase_p_1, phase_n_1, b, step = 0.01, m = mass, g_vec=g_vec, mu = mu)
		print "last step valiud at phase: ", step
		if(step == 0.0):
			break
		if(not found):
			wps = wps[:-1] + [c_ddc] + [wps[-1]]

	
	
	print "found? ", found
	
	if found:
		
		print "plot trajectory"
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		n = 100
		points = [b(0.01 * i)[0] for i in range(100)]
		xs = [point[0] for point in points]
		ys = [point[1] for point in points]
		zs = [point[2] for point in points]
		ax.scatter(xs, ys, zs, c='b')

		colors = ["r", "b", "g"]
		#~ #print contact points of first phase
		xs = [point[0] for point in phase_p_3]
		ys = [point[1] for point in phase_p_3]
		zs = [point[2] for point in phase_p_3]
		ax.scatter(xs, ys, zs, c=colors[0])
			
		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')
		
		#~ #print control points
		xs = [point[0] for (point,_) in wps]
		ys = [point[1] for (point,_) in wps]
		zs = [point[2] for (point,_) in wps]
		ax.scatter(xs, ys, zs, c=colors[2])
			
		#now draw control points
			
	#~ plt.show()
	#find two points in the cone
	
	#~ m = 54.
	#~ g_vec=array([0.,0.,-9.81])
	#~ c0_ddc0 = (array([10. for _ in range(3)]), array([1. for _ in range(3)]))
	#~ c1_ddc1 = (array([25. for _ in range(3)]), array([3. for _ in range(3)]))
	#~ b = bezier_traj([c0_ddc0, c1_ddc1])
