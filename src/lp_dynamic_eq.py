"""
Created on Tue March 7

@author: stonneau
"""


from numpy import array, zeros, ones, sqrt, cross, asmatrix, matrix
from centroidal_dynamics_methods import compute_G

NUMBER_TYPE = 'float'  # 'float' or 'fraction'
                

from centroidal_dynamics import *

def dynamic_equilibrium_lp(c, ddc, P, N, mass = 54., mu = 0.3):
	eq = Equilibrium("dyn_eq", mass, 4) 
	eq.setNewContacts(asmatrix(P),asmatrix(N),mu,EquilibriumAlgorithm.EQUILIBRIUM_ALGORITHM_LP)
	status, robustness = eq.computeEquilibriumRobustness(c, ddc)
	return status == LP_STATUS_OPTIMAL and robustness >= 0., robustness
