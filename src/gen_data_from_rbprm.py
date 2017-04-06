"""
Created on april 6 2017

@author:  stonneau
"""

import sys
sys.path.insert(0, './tools')

from transformations import rotation_matrix, identity_matrix
from numpy import array, cross, zeros, matrix, asmatrix, asarray
from numpy.linalg import norm
import numpy as np
import math
from numpy.random import rand

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
	#first compute com #acceleration always assumed to be 0 and vel as wel	
	Ps, Ns = fullBody.computeContactPoints(stateid)
	#~ states = [make_state(fullBody, Ps[i], Ns[i], configs[stateid+i], viewer) for i in range(0,3,2)]
	states = []
	states += [make_state(fullBody, Ps[0], Ns[0], configs[stateid]  )]
	states += [make_state(fullBody, Ps[2], Ns[2], configs[stateid+1])]
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
	
from bezier_trajectory import connect_two_points
from lp_find_point import find_valid_c_ddc_random
from lp_dynamic_eq import dynamic_equilibrium_lp

flatten = lambda l: [item for sublist in l for item in sublist]

def generate_problem(data, m = 55.88363633, mu = 0.5):
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
	bounds_c = flatten([[min(c0[i], c1[i])-0.1, max(c0[i], c1[i])+0.1] for i in range(3)]) # arbitrary
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
	
	return (c0, ddc0), c_ddc_mid, (c1, ddc1), P_0, N_0, P_1, N_1
	
mu = 0.6
m = 55.88363633
def main(idx = 0):
	all_data = load_data('stair_bauzil_contacts_data')
	c_ddc_0, c_ddc_mid, c_ddc_1, P0, N0, P1, N1 = generate_problem(all_data[idx], m = m, mu = mu)
	connect_two_points(c_ddc_0  , c_ddc_mid, P0, N0, mu = mu, m =  m)
	connect_two_points(c_ddc_mid, c_ddc_1  , P1, N1, mu = mu, m =  m)

if __name__ == '__main__':
	main()
