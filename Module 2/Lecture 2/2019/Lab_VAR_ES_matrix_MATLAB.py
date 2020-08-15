
# 
# LAB: VAR and ES portfolio
# Thierry Roncalli, Introduction to Risk Parity and Budgeting, CRC Press, 2014
# Example page 75
#
# Fitch Learning UK
# Jan 2018

import math
import numpy as np
from scipy.stats import norm


# STEP 1 input data three stocks A,B,C

x = np.matrix([[0.5203], [0.1439], [0.3358]])
mu = np.matrix([[50/10000],[30/10000], [20/10000]])
vol = np.matrix([[2/100], [3/100], [1/100]])

rho = np.matrix([[1, 0.5, 0.25],
[0.5, 1, 0.6],
[0.25, 0.6, 1]])

Sigma = np.matrix([[4.0000e-004, 3.0000e-004, 5.0000e-005],
[3.0000e-004,9.0000e-004, 1.8000e-004],
[5.0000e-005,1.8000e-004,1.0000e-004]])

alpha=0.99


# STEP 2 calculations
# compute VAR
VAR_x = -x.T*mu + norm.ppf(alpha)*math.sqrt(x.T*Sigma*x)

# compute ES
ES_x = -x.T*mu + (math.sqrt(x.T*Sigma*x))/(1-alpha)*norm.pdf(norm.ppf(alpha))


# STEP 3 output results
print('VAR_x  = ', VAR_x )
print('ES_x  = ', ES_x )
