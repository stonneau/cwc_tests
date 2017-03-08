"""
Created on Thurs Aug 4

@author: adelpret, updated by stonneau
"""


from numpy import array, vstack, zeros, sqrt, cross
import numpy as np
from math import cos, sin, tan, atan, pi

from polytope_conversion_utils import *
from centroidal_dynamics_methods import compute_G

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
	G = compute_G(p, N, mu = mu, n =  n ,cg = cg, USE_DIAGONAL_GENERATORS = USE_DIAGONAL_GENERATORS)
	''' convert generators to inequalities '''
	return cone_span_to_face(G, simplify_cones);

