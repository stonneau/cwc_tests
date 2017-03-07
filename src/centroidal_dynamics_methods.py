"""
Created on Thurs Aug 4

@author: adelpret, updated by stonneau
"""

import sys
sys.path.insert(0, './tools')

from polytope_conversion_utils import *
from transformations import euler_matrix
from numpy import array, vstack, zeros, sqrt, cross
import numpy as np

from math import cos, sin, tan, atan, pi
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import axes
from mpl_toolkits.mplot3d import Axes3D
                     
CONTACT_SET = 1;

## 
#  Given a list of contact points
#  as well as a list of associated normals
#  compute contact forces generators V such that 
#  for any beta >0, f = V beta are forces that respect the contact constraints
#  \param p array of 3d contact positions
#  \param N array of 3d contact normals
#  \param mu friction coefficient
#  \param n generator size
#  \param cg number of generators per contact
#  \return the V
def compute_contact_generators(p, N, mu = 0.3, n = 3 ,cg = 4, USE_DIAGONAL_GENERATORS = True):
	''' compute generators '''
	c = p.shape[0];
	#gamma = atan(mu);   # half friction cone angle
	m = c*cg;            # number of generators
	S = np.zeros((n,m));
	T1 = np.zeros((c,n));
	T2 = np.zeros((c,n));
	muu = mu/sqrt(2);
	for i in range(c):
		''' compute tangent directions '''
		N[i,:]  = N[i,:]/np.linalg.norm(N[i,:]);
		T1[i,:] = np.cross(N[i,:], [0,1,0]);
		if(np.linalg.norm(T1[i,:])<1e-5):
			T1[i,:] = np.cross(N[i,:], [1,0,0]);
		T1[i,:] = T1[i,:]/np.linalg.norm(T1[i,:]);
		T2[i,:] = np.cross(N[i,:], T1[i,:]);
		T2[i,:] = T2[i,:]/np.linalg.norm(T2[i,:]);
		
		if(USE_DIAGONAL_GENERATORS):
			S[:,cg*i+0] =  muu*T1[i,:] + muu*T2[i,:] + N[i,:];
			S[:,cg*i+1] =  muu*T1[i,:] - muu*T2[i,:] + N[i,:];
			S[:,cg*i+2] = -muu*T1[i,:] + muu*T2[i,:] + N[i,:];
			S[:,cg*i+3] = -muu*T1[i,:] - muu*T2[i,:] + N[i,:];
		else:
			S[:,cg*i+0] =   mu*T1[i,:] + N[i,:];
			S[:,cg*i+1] =  -mu*T1[i,:] + N[i,:];
			S[:,cg*i+2] =   mu*T2[i,:] + N[i,:];
			S[:,cg*i+3] = - mu*T2[i,:] + N[i,:];
		
		S[:,cg*i+0] = S[:,cg*i+0]/np.linalg.norm(S[:,cg*i+0]);
		S[:,cg*i+1] = S[:,cg*i+1]/np.linalg.norm(S[:,cg*i+1]);
		S[:,cg*i+2] = S[:,cg*i+2]/np.linalg.norm(S[:,cg*i+2]);
		S[:,cg*i+3] = S[:,cg*i+3]/np.linalg.norm(S[:,cg*i+3]);
		
	return S
	

## 
#  Given an aray of contact points
#  ''' compute matrix mapping contact forces to gravito-inertial wrench '''
#  \param p array of contact points
#  \return M the matrix mapping contact forces to gravito-inertial wrench
def compute_contact_to_cwc_matrix(p):	
	c = p.shape[0];
	M = np.zeros((6,3*c));
	for i in range(c):
		M[:3, 3*i:3*i+3] = -np.identity(3);
		M[3:, 3*i:3*i+3] = -crossMatrix(p[i,:]);
	return M
	


