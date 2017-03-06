"""
Created on march 6 2017

@author:  stonneau
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from compute_CWC import test_static_eq
			
def __gen_values(min_point, max_point, inc):
	assert(max_point > min_point)
	num_points = int(float(max_point - min_point) / float(inc))
	return [min_point + inc * v for v in range(num_points)]
        
def __plot_3d_points(ax, points, c = 'b'):
	xs = [point[0] for point in points]
	ys = [point[1] for point in points]
	zs = [point[2] for point in points]
	ax.scatter(xs, ys, zs, c=c)		
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
              
## 
#  Given a computed CWC, plots the valid
#  center of mass positions assuming 0 acceleration.
#  as well as a list of associated normals
#  compute the gravito inertial wrench cone
#  \param H, the CWC, Hw <= 0
#  \param m, mass of the robot
#  \param boundaries on the tested com area, [x_min,x_max,y_min,y_max,z_min,z_max]
#  \param discretizationSteps increments in each direction [x_step,y_step,z_step]
#  redundancies
#  \param params requires "mass" "g"  and "mu"
#  \return the CWC H, H w <= 0, where w is the wrench
def plot_quasi_static_feasible_c(H, m, bounds,discretizationSteps = [0.1,0.1,0.1], P = []):
	x_vals = __gen_values(bounds[0],bounds[1],discretizationSteps[0])
	y_vals = __gen_values(bounds[2],bounds[3],discretizationSteps[1])
	z_vals = __gen_values(bounds[4],bounds[5],discretizationSteps[2])
	valid_points =  [[x_i,y_i,z_i] for  x_i in x_vals for  y_i in y_vals for  z_i in z_vals if test_static_eq(H,np.array([x_i,y_i,z_i]),m)]	
	fig = plt.figure()	
	ax = fig.add_subplot(111, projection='3d')
	__plot_3d_points(ax, P, 'r') #contact points
	__plot_3d_points(ax, valid_points)
	plt.show()     
	return valid_points

