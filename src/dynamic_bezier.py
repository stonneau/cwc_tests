"""
Created on march 6 2017

@author:  stonneau
"""

import sys
sys.path.insert(0, './tools')

from transformations import rotation_matrix, identity_matrix
from numpy import array, cross, zeros, matrix, asmatrix, asarray, vstack, hstack
from numpy.linalg import norm
import numpy as np
import math
from lp_dynamic_eq import dynamic_equilibrium_lp
from lp_find_point import find_valid_c_cwc, find_valid_ddc_cwc, find_valid_c_ddc_cwc, find_valid_c_ddc_random, find_valid_c_random, qp_ineq_6D_orthogonal, qp_ineq_3D_line

from CWC_methods import compute_w, compute_CWC, is_stable

#importing bezier routines
from spline import bezier, curve_constraints, bezier6

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
	
def bezier_as_array(b):
	def res(t):
		return asarray(b(t)).flatten()
		
def eval_as_array(b, t):
	return asarray(b(t)).flatten()
	
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
	return __compute_c_ddc_t(c_t), c_t
	

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


# dynamic stuff
def get_w_of_t(c_t, m = 54., g_vec=array([0.,0.,-9.81])):
	#first get ddc
	ddc_t = c_t.compute_derivate(2)
	def w_of_t(t):
		m_cdd = asarray(m * (ddc_t(t)  - matrix(g_vec).transpose())).flatten()
		return hstack([m_cdd, cross(asarray(c_t(t)).flatten(), m_cdd)])
	return w_of_t
	
	
def get_c_of_t_from_w(w_of_t, c_t):
	w_t = w_of_t(0.3)
	m_w1 = -w_t[0:3]
	c0w =  cross(w_t[3:],m_w1)/(m_w1.dot(m_w1))
	# solve for t = alpla
	c0 = c_t(0.5)
	alpha = ((c0[2] - c0w[2]) /  m_w1[2])[0,0]
	print "*********************** alpha ***************", alpha
	def res(t):
		w_t = w_of_t(t)
		m_w1 = -w_t[0:3]
		return cross(w_t[3:],m_w1)/(m_w1.dot(m_w1)) + alpha * m_w1
	return res
	

import matplotlib
#~ matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

g_vec=array([0.,0.,-9.81])

if __name__ == '__main__':
	
	
	from numpy import identity
	#importing test contacts
	from contact_data_test import *
	g_vec=array([0.,0.,-9.81])
	[c0_ddc0, success, margin] = find_valid_c_random(phase_p_2, phase_n_2, m = mass, mu = mu)
	[c1_ddc1, success, margin] = find_valid_c_random(phase_p_2, phase_n_2, m = mass, mu = mu)
	b, c_t = bezier_traj([c0_ddc0, c1_ddc1], init_dc_ddc = (zero3,c0_ddc0[1]), end_dc_ddc = (zero3,c1_ddc1[1]))
	
	def go(c_t):
	
		w_t = get_w_of_t(c_t)
		#~ print "c(0)\n" + str(c_t(0))
		#~ print "c(1)\n" + str(c_t(1))
		#~ print "w(0)\n" + str(w_t(0))
		#~ print "w(1)\n" + str(w_t(1))
		H = H2
		h = zeros(H.shape[0])
		print "assert o,ot values are ok"
		assert ((H.dot(w_t(0.))<=0.01).all())
		assert ((H.dot(w_t(1.))<=0.01).all())
		
		c_of_t_from_w = get_c_of_t_from_w(w_t,c_t)
		
		print "initial w ", w_t(0.6)
		success, x_proj = qp_ineq_6D_orthogonal(w_t(0.6), H, h)
		print "found", x_proj
		
		#check that found point is feasible
		print "check that found point is feasible ", (H.dot(x_proj)<=0.01).all()
		
		P = zeros([6,6]); P[3:6,:3] = identity(3)
		print "check that w1 and w2 are indeed orthogonal ", norm(x_proj.transpose().dot(P).dot(x_proj)**2) < 0.001
		
		print "create bezier curves using  found point, and initial and final points ... "
		waypoints = matrix([ w_t(0), x_proj, w_t(1) ]).transpose()
		w_t_proj = bezier6(waypoints)
		
		print "check that initial points is reachd ", norm( asarray(w_t_proj(0)).flatten() - w_t(0)) < 0.001
		print "check that final point is reachd ",  norm( asarray(w_t_proj(1.)).flatten() - w_t(1)) < 0.001
		
		print "check that all values in between are such that w1 and w2 are orthogonal AND within H"
		safe = True
		for i in range(100):
			w_i = asarray(w_t_proj(i*0.01)).flatten()
			if not (H.dot(w_i)<=0.01).all():
				print "fail at stability check at value ", i,  " ",  H.dot(w_i).max()
				safe = False
				break 
			if not (norm(x_proj.transpose().dot(P).dot(x_proj)**2) < 0.001):
				print "fail at orthogonality check at value ", i , " " , norm(x_proj.transpose().dot(P).dot(x_proj)**2)
				safe = False
				break
		
		if safe:
			print "w_t_proj is sane !"
			
		print "reconstructing c of t"
		
		w_ref = asarray(w_t_proj(0.)).flatten()
		c_ref = asarray(c_t(0.)).flatten()
		
		print "first c ", c_ref
		"solving for closest c with w_ref"
		res, c_new = qp_ineq_3D_line(c_ref, w_ref)
		
		print "asserting that first c is equal to c_ref", norm(c_ref - c_new) <= 0.001
		
		w_ref = asarray(w_t_proj(1.)).flatten()
		c_ref = asarray(c_t(1.)).flatten()
		
		print "last c ", c_ref
		"solving for closest c with w_ref"
		res, c_new = qp_ineq_3D_line(c_ref, w_ref)
		print "asserting that last c is equal to c_ref", norm(c_ref - c_new) <= 0.001
		
		#reconstructing c(t)
		
		def _gen_approx(f, n, constraints):		
			nf = float(n)
			waypoints = matrix([f(float(i) / nf) for i in range(n+1)]).transpose()
			if constraints == None:
				return bezier(waypoints)
			else:
				return bezier(waypoints, constraints)
		
		def reconstruct_c_at(w_t, c_t):
			def closure(t):
				w_ref = asarray(w_t(t)).flatten()
				c_ref = asarray(c_t(t)).flatten()
				res, c_new = qp_ineq_3D_line(c_ref, w_ref)
				if not res:
					print "qp failed"
				#~ print "res qp at ", t, " ", c_new, " norm: ", norm(c_ref - c_new) 
				return c_new
			return closure
		
		def reconstruct_ddc_at(w_t, m = 54., g_vec=array([0.,0.,-9.81])):
			def closure(t):
				return asarray(w_t(t)).flatten()[0:3] / m + g_vec
			return closure
				
		def _gen_bezier(w_t_proj, c_t, n, constraints = None):
			f = reconstruct_c_at(w_t_proj, c_t)
			return _gen_approx(f, n+ 1, constraints)
			
		def _ddc_t_from_wt(w_t_proj, n):
			f = reconstruct_ddc_at(w_t_proj)
			return _gen_approx(f, n+ 1, constraints=None)
			
		def _c_t_from_ddc(w_t_proj, c_t, n, constraints = None):
			ddc_b = _ddc_t_from_wt(w_t_proj, n)
			return ddc_b.compute_primitive(2)
			
		
		print "reconstructing c(t) with 1, 3, 5, 7  waypoint "
		res_1 = _gen_bezier(w_t_proj, c_t, 1)
		res_3 = _gen_bezier(w_t_proj, c_t, 3)
		res_5 = _gen_bezier(w_t_proj, c_t, 5)
		res_7 = _gen_bezier(w_t_proj, c_t, 7)
		
		print "reconstructing c(t) with 1, 3, 5, 7  waypoint and constraints "
		c = curve_constraints();
		res_1_c = _c_t_from_ddc(w_t_proj, c_t, 1, c)
		res_3_c = _c_t_from_ddc(w_t_proj, c_t, 3, c)
		res_5_c = _c_t_from_ddc(w_t_proj, c_t, 5, c)
		res_7_c = _c_t_from_ddc(w_t_proj, c_t, 7, c)
		#~ res_1_c = _gen_bezier(w_t_proj, c_t, 1, c)
		#~ res_3_c = _gen_bezier(w_t_proj, c_t, 3, c)
		#~ res_5_c = _gen_bezier(w_t_proj, c_t, 5, c)
		#~ res_7_c = _gen_bezier(w_t_proj, c_t, 7, c)
		
		print "now checking if the trajectories are okay"
		def check(curve):
			wt = get_w_of_t(curve, m = 54., g_vec=array([0.,0.,-9.81]))		
			nb_succ = 100
			for i in range(100):
				if not (H.dot(wt(i * 0.01))<=0.01).all():
					#~ print "stability check failed at ", i, H.dot(wt(i * 0.01)).max()
					nb_succ -= 1 
			#~ print "stability check success"
			return nb_succ
		
		success = [check(x) for x in [c_t, res_1, res_3, res_5, res_7, res_1_c, res_3_c, res_5_c, res_7_c]]
		for i, val in enumerate(success):
			print "res for curve ", i , " : ", val
		
		import matplotlib.cm as cm
		colors = cm.rainbow(np.linspace(0, 1, 10))
		
		#plot c_t	
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		n = 100
		
		points = [eval_as_array(c_t, 0.01 * i) for i in range(100)]
		xs = [point[0] for point in points]
		ys = [point[1] for point in points]
		zs = [point[2] for point in points]
		ax.scatter(xs, ys, zs, c='r')
		
		points = [eval_as_array(res_1, 0.01 * i) for i in range(100)]
		xs = [point[0] for point in points]
		ys = [point[1] for point in points]
		zs = [point[2] for point in points]
		#~ ax.scatter(xs, ys, zs, c=colors[1])
		
		#plot ddc_t	
		#~ points = [eval_as_array(res_3, 0.01 * i) for i in range(100)]
		#~ xs = [point[0] for point in points]
		#~ ys = [point[1] for point in points]
		#~ zs = [point[2] for point in points]
		#~ ax.scatter(xs, ys, zs, c=colors[2])
		#~ 
		#~ 
		#plot c_of_w_t	
		#~ points = [eval_as_array(res_5, 0.01 * i) for i in range(100)]
		#~ xs = [point[0] for point in points]
		#~ ys = [point[1] for point in points]
		#~ zs = [point[2] for point in points]
		#~ ax.scatter(xs, ys, zs, c=colors[3])
		
		#~ #plot c_of_w_t	
		points = [eval_as_array(res_7, 0.01 * i) for i in range(100)]
		xs = [point[0] for point in points]
		ys = [point[1] for point in points]
		zs = [point[2] for point in points]
		ax.scatter(xs, ys, zs, c='g')
		
		points = [eval_as_array(res_1_c, 0.01 * i) for i in range(100)]
		xs = [point[0] for point in points]
		ys = [point[1] for point in points]
		zs = [point[2] for point in points]
		#~ ax.scatter(xs, ys, zs, c=colors[5])
		
		#plot ddc_t	
		#~ points = [eval_as_array(res_3, 0.01 * i) for i in range(100)]
		#~ xs = [point[0] for point in points]
		#~ ys = [point[1] for point in points]
		#~ zs = [point[2] for point in points]
		#~ ax.scatter(xs, ys, zs, c=colors[6])
		#~ 
		#~ 
		#plot c_of_w_t	
		#~ points = [eval_as_array(res_5, 0.01 * i) for i in range(100)]
		#~ xs = [point[0] for point in points]
		#~ ys = [point[1] for point in points]
		#~ zs = [point[2] for point in points]
		#~ ax.scatter(xs, ys, zs, c=colors[7])
		
		#~ #plot c_of_w_t	
		points = [eval_as_array(res_7_c, 0.01 * i) for i in range(100)]
		xs = [point[0] for point in points]
		ys = [point[1] for point in points]
		zs = [point[2] for point in points]
		ax.scatter(xs, ys, zs, c='b')
		
		return res_5_c
	#~ 
	#~ plt.show()
	c = go(c_t)
	
def old_main():
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
