"""
Created on Tue March 7

@author: stonneau
"""


from numpy import array, zeros, ones, sqrt, cross
from centroidal_dynamics_methods import compute_contact_generators, compute_contact_to_cwc_matrix

NUMBER_TYPE = 'float'  # 'float' or 'fraction'

g_vec = array([0,0,-9.81])
                
def __compute_G(p, N, mu):
	V = compute_contact_generators(p, N, mu = mu)
	p_cross = compute_contact_to_cwc_matrix(p)	
	print "p_cross", p_cross.shape
	print "V", V.shape
	print "V", V
	return p_cross.dot(V)

def __compute_h(c,mass):
	h = np.zeros(6);
	h[0:3] =  -g_vec
	h[3:6] =  cross(c, -g_vec)
	return mass * h
		
def __compute_H_ddc(c,ddc,mass):
	H_ddc = np.zeros(6);
	h[0:3] =  ddc
	h[3:6] =  cross(c, ddc)
	return mass * h
		
                     
	from scipy.optimize import linprog
 
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
	G = __compute_G(p, N, mu);
	h = __compute_h(c, mass)
	H_ddc = __compute_H_ddc(c,ddc,mass)
	size_beta = G.shape[1]
	cost = ones(size_beta)
	#define constraint beta >= 0, and beta not constrainted
	bounds = tuple([(0, None) for _ in cost])
	res = linprog(cost, A_eq=G, b_eq=H_ddc + h, bounds=bounds,options={"disp": True})
