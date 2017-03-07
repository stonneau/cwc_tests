"""
Created on Thurs Aug 4

@author: adelpret, updated by stonneau
"""


from numpy import array, vstack, zeros, sqrt, cross
import numpy as np
from math import cos, sin, tan, atan, pi

from polytope_conversion_utils import *
from centroidal_dynamics_methods import compute_contact_generators, compute_contact_to_cwc_matrix

NUMBER_TYPE = 'float'  # 'float' or 'fraction'

                     

n = 3;      # generator size
cg = 4;     # number of generators per contact
USE_DIAGONAL_GENERATORS = True;
CONTACT_SET = 1;

## 
#  Given a list of contact points
#  as well as a list of associated normals
#  compute the gravito inertial wrench cone
#  \param p array of 3d contact positions
#  \param N array of 3d contact normals
#  \param mu friction coefficient
#  \param simplify_cones if true inequality conversion will try to remove 
#  redundancies
#  \param params requires "mu"
#  \return the CWC H, H w <= 0, where w is the wrench
def compute_CWC(p, N, mu = 0.3, simplify_cones = False):
	S = compute_contact_generators(p, N, mu = mu, n =  n ,cg = cg, USE_DIAGONAL_GENERATORS = USE_DIAGONAL_GENERATORS)
	M = -compute_contact_to_cwc_matrix(p) #sign reversed compared to paper ICRA 15 from del prete
	c = p.shape[0]
	m = c*cg;            # number of generators
	''' project generators in 6d centroidal space '''
	S_centr = np.zeros((6,m));
	for i in range(c):
		S_centr[:,cg*i:cg*i+cg] = np.dot(M[:,3*i:3*i+3], S[:,cg*i:cg*i+cg]);
	''' convert generators to inequalities '''
	return cone_span_to_face(S_centr, simplify_cones);

