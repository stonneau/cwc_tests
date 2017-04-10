"""
Created on march 6 2017

@author:  stonneau
"""

import sys
sys.path.insert(0, './tools')

from stability import test_eq_cwc
from plot_cond import plot_cond
from transformations import rotation_matrix, identity_matrix
from numpy import array, cross, zeros, ones, identity
from CWC_methods import compute_CWC, is_stable
from lp_dynamic_eq import dynamic_equilibrium_lp
from lp_find_point import find_valid_c_cwc, find_valid_ddc_cwc, find_valid_c_ddc_cwc, find_valid_c_ddc_random

import numpy as np
import math

#importing test contacts
from contact_data_test import *

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
	if(not status):
		print "[ERROR] LP find_intersection_c is not feasible"
		return
		#~ 
	c= wp_1[0:3][:]			
	print "solution, (c / ddc) ", c , " " , ddc, "margin", wp_1[3]
	
	
	_print_res(1, H1, c, ddc, phase_p_1, phase_n_1, mass, mu)	
	_print_res(2, H2, c, ddc, phase_p_2, phase_n_2, mass, mu)	
	_print_res(3, H3, c, ddc, phase_p_3, phase_n_3, mass, mu)	
	
	#try to impose negative c
	print "trying to constrain solution between -1 and 0"
	Kin_c = zeros((6,3))
	Kin_c[:3,:] = identity(3)
	#~ Kin_c[3:5,:-1] = -identity(2)
	kin_c = zeros(6)
	kin_c[:3] = ones(3) * 200
	#~ kin_c[3:5] = ones(2)
	
	print "Kin_c, " , Kin_c
	print "kin_c, " , kin_c
	
	
	status, sol_found, wp_1 = find_valid_c_cwc(H, ddc, (Kin_c, kin_c), m = mass)
		
	from pinocchio_inv_dyn.optimization import solver_LP_abstract
	c= wp_1[0:3][:]	
	print "constyrained solution, (c / ddc) ", c , " " , ddc, "margin", wp_1[3]
	
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
	
def test_find_c_ddc_rand(P, N, mu = 0.6):
	#~ [(c,ddc), success, margin] = find_valid_c_ddc(H, ddc=array([ 0.04380291,  0.67393901,  0.7374873 ]), m = mass)	
	[(c,ddc), success, margin] = find_valid_c_ddc_random(P, N, bounds_c = [0.,1.,0.,1.,1.,2.], m = mass, mu = mu)	
	print "solution found ? ", success
	print "Best solution, (c / ddc) ", c , " " , ddc, "margin", margin
	
	
	print "trying to constrain solution between -1 and 0"
	Kin_c = zeros((6,3))
	Kin_c[:3,:] = identity(3)
	#~ Kin_c[3:5,:-1] = -identity(2)
	Kin_c[5,-1] = -1
	kin_c = zeros(6)
	kin_c[:3] = ones(3) 
	kin_c[2] = 3.	
	kin_c[5] = -2
	
	
	print "Kin_c, " , Kin_c
	print "kin_c, " , kin_c
	
	[(c,ddc), success, margin] = find_valid_c_ddc_random(P, N, Kin=(Kin_c, kin_c),bounds_c = [0.,1.,0.,1.,1.,3.], m = mass, mu = mu)
	
	print "solution found ? ", success
	print "Best solution, (c / ddc) ", c , " " , ddc, "margin", margin
		
	_print_res(1, H1, c, ddc, phase_p_1, phase_n_1, mass, mu)	
	_print_res(2, H2, c, ddc, phase_p_2, phase_n_2, mass, mu)	
	_print_res(3, H3, c, ddc, phase_p_3, phase_n_3, mass, mu)	
	
	
#~ print "*********** TEST H1 ********"
#~ test(H1)
print "*********** TEST H2 ********"
#~ test_find_c(H2)
#~ test_find_c_ddc(H2)
test_find_c_ddc_rand(phase_p_2,phase_n_2)
#~ print "*********** TEST H3 ********"
#~ test(H3)
