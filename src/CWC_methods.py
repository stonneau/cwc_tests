"""
Created on Thurs Aug 4

@author: adelpret, updated by stonneau
"""

import sys
sys.path.insert(0, './tools')

from polytope_conversion_utils import *
from transformations import euler_matrix
from numpy import array, vstack, zeros, sqrt, cross, matrix, asmatrix
import numpy as np

from math import cos, sin, tan, atan, pi
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import axes
from mpl_toolkits.mplot3d import Axes3D
                     
CONTACT_SET = 1;
cg = 4;     # number of generators per contact
__EPS = 0.00000001


from centroidal_dynamics import *

#  Compute the centroidal wrench w
#  as well as a list of associated normals
#  compute the G matrix mapping the contact forces generator to the 6D centroidal cone
#  \param c the  COM position
#  \param ddc the COM acceleration
#  \param dL the angular momentum rate
#  \param m mass of the robot
#  \param g_vec gravity vector
#  \return the centroidal wrench w	
def compute_w(c, ddc, dL=array([0.,0.,0.]), m = 54., g_vec=array([0.,0.,-9.81])):
	w1 = m * (ddc - g_vec)
	return array(w1.tolist() + (cross(c, w1) + dL).tolist())
	
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
#  \return the CWC H, H w <= 0, where w is the wrench. WARNING! TODO: The H matrix is such that 
# the wrench w is the one present in the ICRA paper 15 of del prete et al., contrary to the current c++ implementation
def compute_CWC(p, N, mu = 0.3, simplify_cones = False):	
	eq = Equilibrium("dyn_eq2", 54., cg)  #mass unused
	eq.setNewContacts(asmatrix(p),asmatrix(N),mu,EquilibriumAlgorithm.EQUILIBRIUM_ALGORITHM_PP)
	return -eq.getPolytopeInequalities()[0]

## 
#  Given a cone and a wrench returns whether
#  the robot is in equilibrium
#  compute the gravito inertial wrench cone
#  \param H cone
#  \param c COM position
#  \param ddc cone
#  \param H cone
#  \param N array of 3d contact normals
#  \param mu friction coefficient
#  \param simplify_cones if true inequality conversion will try to remove 
#  redundancies
#  \param params requires "mu"
#  \return the CWC H, H w <= 0, where w is the wrench
def is_stable(H,c, ddc=array([0.,0.,0.]), dL=array([0.,0.,0.]), m = 54., g_vec=array([0.,0.,-9.81]), robustness = 0.):
	w = compute_w(c, ddc, dL, m, g_vec)	
	return (H.dot(w)<=-robustness).all()
