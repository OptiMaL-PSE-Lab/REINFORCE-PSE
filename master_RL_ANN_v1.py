# import plotting
from pylab import *
import matplotlib.pyplot as plt
# model integration imports
import numpy as np
import scipy.integrate as scp
import Model_Integrator
from Model_Integrator import ModelIntegration
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
from RL_utilities import PreTraining
from RL_utilities import ComputeRun
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

#final_state = ModelIntegration(params,initial_state_I,contrl,dtime)

runs_PT = 3
inputs_PT = [[i_PT*5.0/t_steps for i_PT in range(t_steps)]] # list of control inputs
states_n = 2
pert_size = 0.1

# pre train to achieve roughly inputs_PT policy (includes perturbance on control)
policy =  PreTraining(policy, params, dtime, tf, states_n, inputs_PT, runs_PT, pert_size)
print(' --- Pre-Trainning Done! ---')
# compute run for single epoc to test
runs_train = 3
control_n = 1
std_sqr = 1.0
output =  ComputeRun(policy, params, dtime, tf, states_n, control_n, t_steps,
 runs_train, std_sqr, initial_state_CR=np.array([1,0]), plot_CR=True)

print('output = ',output)
sys.exit()

# ----- the rest must be deleted ----- #




tj=0
initial_state_P = np.hstack([initial_state_I,tf-tj]) # add time
initial_state_P = Variable(torch.Tensor(initial_state_P))
y1_l = []
y2_l = []
u_l = []
for step_PT_plot in range(t_steps):
    y1_l.append(float(initial_state_I[0]))
    y2_l.append(float(initial_state_I[1]))
    controls=policy(initial_state_P)
    contrl={'U_u':float(controls[0])}
    u_l.append(float(controls[0]))
    final_state = ModelIntegration(params,initial_state_I,contrl,dtime)
    initial_state_I=copy.deepcopy(final_state)
    tj=tj+dtime # calculate next time
    initial_state_P=np.hstack([initial_state_I,tf-tj])
    initial_state_P = Variable(torch.Tensor(initial_state_P)) # make it a torch variable

plt.subplot2grid((2,4),(0,0),colspan=2)
plt.plot(y1_l)
plt.ylabel('y1 ',rotation= 360,fontsize=15)
plt.xlabel('time',fontsize=15)

plt.subplot2grid((2,4),(0,2),colspan=2)
plt.plot(y2_l)
plt.ylabel('y2 ',rotation= 360,fontsize=15)
plt.xlabel('time',fontsize=15)

plt.subplot2grid((2,4),(1,0),colspan=2)
plt.plot(u_l)
plt.ylabel('u',rotation= 360,fontsize=15)
plt.xlabel('time',fontsize=15)
plt.show()
