"""
Created on april 6 2017

@author:  stonneau
"""

import sys
sys.path.insert(0, './tools')

from transformations import rotation_matrix, identity_matrix
from numpy import array, cross, zeros, matrix, asmatrix, asarray
from numpy.random import rand, uniform
from numpy.linalg import norm
import numpy as np
import math
from numpy.random import rand
from CWC_methods import compute_CWC, is_stable
from lp_dynamic_eq import dynamic_equilibrium_lp

zero3 = array([0.,0.,0.])

def make_state_contact_points(P, N):
	res = {}
	res["P"] = P;
	res["N"] = N;
	return res

def make_state(fullBody, P, N, config):
	res = make_state_contact_points(P, N)
	fullBody.client.basic.robot.setCurrentConfig(config)
	res["c"] = array(fullBody.client.basic.robot.getComPosition())
	res["dc"] = zero3
	res["ddc"] = zero3
	return res	

def gen_sequence_data_from_state(fullBody, stateid, configs):	
	print "state id", stateid
	#first compute com #acceleration always assumed to be 0 and vel as wel	
	Ps, Ns = fullBody.computeContactPoints(stateid)
	#~ states = [make_state(fullBody, Ps[i], Ns[i], configs[stateid+i], viewer) for i in range(0,3,2)]
	states = []
	states += [make_state(fullBody, Ps[0], Ns[0], configs[stateid]  )]
	states += [make_state(fullBody, Ps[-1], Ns[-1], configs[stateid+1])]
	return { "start_state" : states[0], "end_state" : states[1], "inter_contacts" : make_state_contact_points(Ps[1], Ns[1]) }
	

def gen_all_sequence_state(fullBody, configs):
	return [gen_sequence_data_from_state(fullBody, i, configs) for i in range(1,len(configs)-3)]
		
def gen_and_save(fullBody, configs, filename):
	from pickle import dump
	res = gen_all_sequence_state(fullBody, configs)
	f= open(filename, "w")	
	dump(res,f)
	f.close()
	
def load_data(filename):
	from pickle import load	
	f= open(filename, "r")	
	res = load(f)
	f.close()
	return res
	
from bezier_trajectory import *
from lp_find_point import find_valid_c_random, find_valid_c_ddc_random
from lp_dynamic_eq import dynamic_equilibrium_lp

flatten = lambda l: [item for sublist in l for item in sublist]

def generate_problem(data, test_quasi_static = False, m = 55.88363633, mu = 0.5):
	# generate a candidate c, ddc valid for the intermediary phase	
	P_mid = data["inter_contacts"]["P"]
	N_mid = data["inter_contacts"]["N"]
	P_0   = data["start_state"]["P"]
	N_0   = data["start_state"]["N"]
	P_1   = data["end_state"]["P"]
	N_1   = data["end_state"]["N"]
	c0 = data["start_state"]["c"]
	dc0 = data["start_state"]["dc"]
	ddc0 = data["start_state"]["ddc"]
	c1 = data["end_state"  ]["c"]
	dc1 = data["end_state"  ]["dc"]
	ddc1 = data["end_state"  ]["ddc"]
	
	#first try to find quasi static solution
	quasi_static_sol = False
	success = False
	bounds_c = flatten([[min(c0[i], c1[i])-0.1, max(c0[i], c1[i])+0.1] for i in range(3)]) # arbitrary
	if test_quasi_static:
		[c_ddc_mid, success, margin] = find_valid_c_random(P_mid, N_mid, bounds_c=bounds_c, m = m, mu = mu)
	if(success):
		quasi_static_sol = True;
	else:
		[c_ddc_mid, success, margin] = find_valid_c_ddc_random(P_mid, N_mid, bounds_c=bounds_c, m = m, mu = mu)
		assert success, "failed to generate valid candidate"	
		res_lp, robustness =  dynamic_equilibrium_lp(c_ddc_mid[0], c_ddc_mid[1], P_mid, N_mid, mass = m, mu = mu)
		assert robustness >= 0., "randome config not equilibrated regarding intermediate contact set " + str (robustness) + str(" ") + str (c_ddc_mid)
		res_lp, robustness =  dynamic_equilibrium_lp(c_ddc_mid[0], c_ddc_mid[1], P_0, N_0, mass = m, mu = mu)
		assert robustness >= 0., "randome config not equilibrated regarding first contact set " + str (robustness) + str(" ") + str (c_ddc_mid)
		res_lp, robustness =  dynamic_equilibrium_lp(c_ddc_mid[0], c_ddc_mid[1], P_1, N_1, mass = m, mu = mu)
		assert robustness >= 0., "randome config not equilibrated regarding first contact set " + str (robustness) + str(" ") + str (c_ddc_mid)
		res_lp, robustness =  dynamic_equilibrium_lp(c0, ddc0, P_0, N_0, mass = m, mu = mu)
		assert robustness >= 0., "init config not equilibrated regarding first contact set " + str (robustness) + str(" ") + str ((c0, ddc0))
		res_lp, robustness =  dynamic_equilibrium_lp(c1, ddc1, P_1, N_1, mass = m, mu = mu)
		assert robustness >= 0., "init config not equilibrated regarding first contact set " + str (robustness) + str(" ") + str ((c1, ddc1))
	
	return quasi_static_sol, (c0, ddc0), c_ddc_mid, (c1, ddc1), P_0, N_0, P_1, N_1




results = { "num_quasi_static" : 0, "trials_fail" :[], "trials_success" :[] }

def saveTrial(c_ddc_0  , c_ddc_1, P, N, success, K):
	global results
	if(success):
		results['trials_success'] += [(c_ddc_0 , c_ddc_1, P, N, K)]
	else:
		results['trials_fail']    += [(c_ddc_0 , c_ddc_1, P, N, K)]

mu = 0.6
m = 55.88363633
def gen_non_trivial_data(idx = 0, num_iters = 100, plt =False):
	all_data = load_data('stair_bauzil_contacts_data')
	quasi_static_sol, c_ddc_0, c_ddc_mid, c_ddc_1, P0, N0, P1, N1 = generate_problem(all_data[idx], test_quasi_static=False, m = m, mu = mu)
	K0 = compute_CWC(P0, N0, mass=m, mu = mu, simplify_cones = False)
	K1 = compute_CWC(P1, N1, mass=m, mu = mu, simplify_cones = False)
	for i in range(num_iters):		
		quasi_static_sol, c_ddc_0, c_ddc_mid, c_ddc_1, P0, N0, P1, N1 = generate_problem(all_data[idx], test_quasi_static=i==0, m = m, mu = mu)
		if(quasi_static_sol):
			global results
			results['num_quasi_static'] += 1			
			print "quasi static solution, "
			#~ return  c_ddc_mid
		#~ else:
		
		found, init_traj_ok = connect_two_points(c_ddc_0  , c_ddc_mid, P0, N0, mu = mu, m =  m,use_cone_for_eq = K0, plot = plt)
		if(not init_traj_ok):
			saveTrial(c_ddc_0  , c_ddc_mid, P0, N0, found, K0)
		found, init_traj_ok = connect_two_points(c_ddc_mid, c_ddc_1  , P1, N1, mu = mu, m =  m,use_cone_for_eq = K1, plot = plt)
		if(not init_traj_ok):
			saveTrial(c_ddc_mid, c_ddc_1  , P1, N1, found, K1)
	print "num success, ", len(results['trials_success'])
	print "num success, ", len(results['trials_success'])
	print "num fails, ", len(results['trials_fail'] )
	print "num quasi static , ", results["num_quasi_static"]
	results['instant_successes'] = num_iters - (len(results['trials_success']) + len(results['trials_fail']))
	print "num additional instant successes", results['instant_successes']


zero3 = array([0.,0.,0.])

def gen_random_waypoint(c0, c1):
	print "c0", c0	
	bounds_c = flatten([[min(c0[i], c1[i])-0.1, max(c0[i], c1[i])+0.1] for i in range(3)]) # arbitrary
	rand_c = array([uniform(bounds_c[2*i], bounds_c[2*i+1]) for i in range(3)])
	return (rand_c,zero3 ) #acceleration not considered

def gen_random_traj(c_ddc_0 , c_ddc_1, P, N):
	for num_waypoints in range(1,4):
		for num_iter in range(1000):
			waypoints = [c_ddc_0] + [gen_random_waypoint(c_ddc_0[0], c_ddc_1[0]) for _ in range(num_waypoints)] + [c_ddc_1]
			b = bezier_traj(waypoints, init_dc_ddc = c_ddc_0, end_dc_ddc = c_ddc_1)
			found, c_ddc, step = eval_valid_part(P, N, b, step = 0.01, m = m, g_vec=g_vec, mu = mu)
			if(found):
				print "found !"
				return True
	return False

from cwc import OptimError, cone_optimization

def check_feasibility(c_ddc_0, c_ddc_1, P, N, cone):
	x_input = [c_ddc_0[0].tolist() + c_ddc_0[1].tolist(), c_ddc_1[0].tolist() + c_ddc_1[1].tolist()]	
	try:
		#~ cone = compute_CWC(P, N, mass=m, mu = mu, simplify_cones = False)
		assert is_stable(cone,c_ddc_0[0],c_ddc_0[1], m = m ),"is stable 0 , "
		assert is_stable(cone,c_ddc_1[0],c_ddc_1[1], m = m ),"is stable 1 , "
		var_final, params = cone_optimization([P for _ in range(3)], [N for _ in range(3)], x_input, [0.3, 0.7, 1.], 0.1, mu =mu, mass = m, cones = [cone for _ in range(3)], simplify_cones = False, verbose=True)
		
	except:
		print "OPTIM FAILED: not feasible"
		return False#check result is feasiblt
	cs = var_final ['c'] 
	ddcs = var_final ['ddc']
	dLs = var_final ['dL']
	for i in range(len(cs)):
		c = cs[i]; ddc = ddcs[i]
		res_lp, robustness =  dynamic_equilibrium_lp(c, ddc, P, N, mass = m, mu = mu)
		print "robustness ", robustness
		cone_stable = is_stable(cone,c,ddc, m = m )
		print "cone stable ", cone_stable
		if not cone_stable:
			print "dL", dLs[i]
			print "result from optim not feasible"
			return False
	print "feasible"
	return True
			
def gen_trajs(res,gen_random=True):
	successes = 0
	fails = 0
	#~ for (c_ddc_0 , c_ddc_1, P, N) in results['trials_fail']+results['trials_success']:
	#~ cone = None
	for (c_ddc_0 , c_ddc_1, P, N, cone) in results['trials_success']:
	#~ for (c_ddc_0 , c_ddc_1, P, N, cone) in results['trials_fail']:
		#~ if cone == None:
			#~ cone = compute_CWC(P, N, mass=m, mu = mu, simplify_cones = False)
		if gen_random:
			if  gen_random_traj(c_ddc_0 , c_ddc_1, P, N):
				successes+=1
				#try to see why it failed
			else:
				fails+=1
		else:
			#~ found, init_traj_ok = connect_two_points(c_ddc_0  , c_ddc_1, P, N, mu = mu, m =  m, plot = False)	
			found = check_feasibility(c_ddc_0, c_ddc_1, P, N, cone)
			if  found:
				successes+=1
				#~ found, init_traj_ok = connect_two_points(c_ddc_0  , c_ddc_1, P, N, mu = mu, m =  m,use_cone_for_eq = None, plot = False, rob = -2.)
				#~ if found:
					#~ print "success also for bezier"
					#~ successes-=1
				#~ else:
					#~ print "still failed"
			else:
				fails+=1
			
	print "num successes in generating random trajectory: ", successes
	print "num failures in generating random trajectory: ", fails

def save_results(fname ="test"):
	from pickle import dump
	f = open(fname, "w+")
	dump(results, f)
	f.close()

def load_results(fname ="test"):
	from pickle import load
	f = open(fname, "r+")
	res = load (f)
	f.close()
	
	print "num success, ", len(res['trials_success'])
	print "num fails, ", len(res['trials_fail'] )
	print "num additional instant successes", (res['instant_successes'])
	return res

if __name__ == '__main__':
	#~ gen_non_trivial_data(3)
	#~ gen_non_trivial_data(2)
	#~ gen_non_trivial_data(1)
	#~ save_results()
	
	results = load_results()
	gen_trajs(results, False)
