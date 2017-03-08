"""
Created on Tue March 7

@author: stonneau
"""


from numpy import array, zeros, ones, sqrt, cross
from centroidal_dynamics_methods import compute_G

NUMBER_TYPE = 'float'  # 'float' or 'fraction'

g_vec = array([0,0,-9.81])
                
def __compute_h(c,mass):
	h = zeros(6);
	h[0:3] =  -g_vec
	h[3:6] =  cross(c, -g_vec)
	return mass * h
		
def __compute_H_ddc(c,ddc,mass):
	H_ddc = zeros(6);
	H_ddc[0:3] =  ddc
	H_ddc[3:6] =  cross(c, ddc)
	return mass * H_ddc
		

#~ from scipy.optimize import linprog
from pinocchio_inv_dyn.optimization import solver_LP_abstract
 
solver = solver_LP_abstract.getNewSolver('qpoases', "dyn_eq", maxIter=1000, maxTime=100.0, useWarmStart=True, verb=0)

# Formulate the dynamic equilibrium problem as a linear program
# see IROS 17 submission by P. Fernbach et al.
# Find B
# s.t. GB = H ddc + h
# B >= 0
# where ddc is a given acceleration
# G is the force generators projected into the gravito inertial cone
# B is a variable for the forces of dim 6*num_contacts
# h is the matrix m*[[-g];[c X -g]]
# H is m * [[Id_3][c^]]  with c^ the cross product matrix
# if B is found then the system is in dynamic equilibrium
#
#  \param p array of 3d contact positions
#  \param N array of 3d contact normals
#  \param mu friction coefficient
#  \param c COM (array)
#  \param ddc COM  acceleration (array)
#  \return whether the system is in dynamic equilibrium
def dynamic_equilibrium(c, ddc, p, N, mass = 54, mu = 0.3):
	G = compute_G(p, N, mu);
	h = __compute_h(c, mass)
	H_ddc = __compute_H_ddc(c,ddc,mass)
	cost = ones(G.shape[1])
	bounds = tuple([(0, None) for _ in cost])
	lb = array([0. for _ in cost])
	ub = array([10000000. for _ in cost])
	global solver
	(status, res, _) = solver.solve(cost, lb, ub, A_in=None, Alb=None, Aub=None, A_eq=G, b=H_ddc + h)
	return status == solver_LP_abstract.LP_status.OPTIMAL
