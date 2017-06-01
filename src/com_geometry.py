"""
Created on Tue March 7

@author: stonneau
"""

# This file contains method to determine the reachability of contact positions depending
# on existing com constraints, as well as static equilibrium constraints.
# In general, kinematics constraints are enforced, and the solver returns 
# a bound on the static equilibrium value. If the value is positive, the problem 
# is feasible, and a com value is returned.
