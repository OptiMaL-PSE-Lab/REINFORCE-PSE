# import plotting
from pylab import *
import matplotlib.pyplot as plt
# model integration imports
import numpy as np
import scipy.integrate as scp
import Model_Integrator
from Model_Integrator import model_integration
# ANN RL imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn.utils as utils
# from RL utilities
from RL_utilities import PolicyNetwork
from RL_utilities import pretraining
from RL_utilities import compute_run
# import others
import sys
import copy
# random seeds
np.random.seed(seed=666)
torch.manual_seed(666)

contrl={'U_u':0.5}
''' ### definition of model and integration parameters - start  ### '''
params={'a_p' : 0.5, 'b_p' : 1}
initial_state_I = np.array([1,0])
dtime=0.1
t_steps = 10
tf = dtime*t_steps
''' ### ---------------------------------------------- - end  ### '''

''' define policy network '''
policy = PolicyNetwork(6, 3, 1)

#final_state = model_integration(params,initial_state_I,contrl,dtime)

runs_PT = 3
inputs_PT = [[i_PT*5.0/t_steps for i_PT in range(t_steps)]] # list of control inputs
states_n = 2
pert_size = 0.1

# pre train to achieve roughly inputs_PT policy (includes perturbance on control)
policy =  pretraining(policy, params, dtime, tf, states_n, inputs_PT, runs_PT, pert_size)
print(' --- Pre-Trainning Done! ---')
# compute run for single epoc to test
runs_train = 3
control_n = 1
std_sqr = 1.0
output =  compute_run(policy, params, dtime, tf, states_n, control_n, t_steps,
 runs_train, std_sqr, initial_state_CR=np.array([1,0]), plot_CR=True)

print('output = ',output)
sys.exit()
