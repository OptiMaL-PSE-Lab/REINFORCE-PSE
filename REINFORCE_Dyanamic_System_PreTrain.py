import pylab
import numpy as np
import scipy.integrate as scp
from pylab import *
import matplotlib.pyplot as plt
np.random.seed(seed=666)
'''
Here we include multiple episodes per update
We include a *5 on control output to scale it

'''
#secondary utilities
import csv
#import itertools
import os
import sys
import copy
import Model_Integrator
# --- packages for ANNs --- #
import math
import random
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn.utils as utils
# defino some torch stuff
torch.manual_seed(666)
pi = Variable(torch.FloatTensor([math.pi]))

#import torchvision.transforms as T
#---------- Policy network (start) ----------#
'''
Policy-network
Our model will be a feed-forward neural network that takes in the states.
Outputs mean and std of an action (exploraion-explotation).
'''
#number of states + time
class PolicyNetwork(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(PolicyNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size, bias=True)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear3_m = nn.Linear(hidden_size, num_outputs, bias=True)
        self.linear3_sigma = nn.Linear(hidden_size, num_outputs, bias=True)
# ADD relus!!!!
    def forward(self, inputs):
        x = inputs
        #x = F.sigmoid(self.linear1(x))
        #x = (F.sigmoid(self.linear2(x)))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # sigmoid restricts output (0-1), 5 restricts (0-5)
        mu = F.relu6(self.linear3_m(x))
        return mu
'''
----------------------------------------------------
Function to pre-train policy -start
'''
#Pre-Trainning
    # PreTraining(policy, inputs_PT, runs_PT,pert_size, initial_state_I=np.array([1,0])) 
def PreTraining(policy_PT, inputs, runs_PT, pert_size, initial_state_I=np.array([1,0])):
    ''' define lists to be filled '''
    y1_PT = [[None for i_PT in range(t_steps)]  for i_PT in range(runs_PT)]
    y2_PT = [[None for i_PT in range(t_steps)]  for i_PT in range(runs_PT)]
    t_PT = [[None for i_PT in range(t_steps)]  for i_PT in range(runs_PT)]
    U_u_PT = [[None for i_PT in range(t_steps)]  for i_PT in range(runs_PT)]
    for i_episode in range(runs_PT):
        tj=np.array([ti]) # define initial time at each episode
        for step_j in range(t_steps):
            controls=inputs[step_j]*(1+np.random.uniform(-pert_size,pert_size))
            action = controls
            contrl={'U_u':float64(action)}
            final_state = Model_Integrator.ModelIntegration(params,initial_state_I,contrl,dtime)
            initial_state_I=copy.deepcopy(final_state)
            tj=tj+dtime # calculate next time
            y1_PT[i_episode][step_j] = final_state[0]
            y2_PT[i_episode][step_j] = final_state[1]
            t_PT[i_episode][step_j] = tf-tj
            U_u_PT[i_episode][step_j] = float64(action)
    # setting data for trainning
    y_data = [[(U_u_PT[j][i]) for i in range(0,len(U_u_PT[j]))] for j in range(0,len(U_u_PT))]
    x_data = [[(y1_PT[j][i],y2_PT[j][i],t_PT[j][i])
     for i in range(0,len(y1_PT[j]))]
      for j in range(0,len(y1_PT))]
    # passing data as torh vectors
    inputs_l = [Variable(torch.Tensor(x_data[i])) for i in range(0,len(x_data))]
    labels_l = [Variable(torch.Tensor(y_data[j])) for j in range(0,len(y_data))]
    # training parameters
    criterion = nn.MSELoss()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
#    optimizer = torch.optim.LBFGS(policy_PT.parameters(), history_size=10000)
#    def closure():
#        return PT_loss
    epoch_n = 400
    # start training
    for PT_epoch in range(epoch_n):
        optimizer.zero_grad()
        PT_loss = 0
        for kk in range(len(inputs_l)):
            for inpt, label in zip(inputs_l[kk], labels_l[kk]):
                output = policy_PT(inpt)
                PT_loss += criterion(output, label)
        sys.stdout.write("predicted string: ")
        print(", epoch: %d, loss: %1.3f" % (PT_epoch + 1, PT_loss.data[0]))
        PT_loss.backward()
#        optimizer.step(closure)
        optimizer.step()
    return y1_PT[0], y2_PT[0], t_PT[0], U_u_PT[0]

'''
----------------------------------------------------
Function to pre-train policy -end
'''

def normal(act, mu, sigma_sq):
    a = (-1*(Variable(act)-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/np.sqrt((2*sigma_sq*pi))
    return a*b
'''
select_action: will select an action accordingly to an epsilon greedy policy.
In the constinous space, this means adding a random perturbation to our control
np.random.normal: Draw random samples from a normal (Gaussian) distribution.
>>> mu, sigma = 0, 0.1 # mean and standard deviation
>>> s = np.random.normal(mu, sigma, 1000)
'''
# prints for 2 controls
def select_action(control_mean, control_sigma, train=True):
    # control_sigma =  tensor([ 0.4420,  0.3498])
    # control_mean =  tensor([ 2.9110,  2.0363])
    if train==True:
        eps = torch.FloatTensor([torch.randn(control_mean.size())])
        # control_sigma =  tensor([ 0.4420,  0.3498])
        control_choice = (control_mean + np.sqrt(control_sigma)*Variable(eps)).data
        # control_choice =  tensor([ 3.2173,  2.0184])
        prob = normal(control_choice, control_mean, control_sigma)
        # prob =  tensor([ 0.5396,  0.6742])
        log_prob = prob.log()
        entropy = -0.5*((control_sigma+2*pi).log()+1)
        return control_choice, log_prob, entropy
    elif train==False:
        return control_mean
'''
problem parameters
'''
#memory = ReplayMemory(10000)
episode_n = 5000000 # define number of episodes !!!
record_n = 10000 # !!!
std_sqr_red = (1,record_n) # !!!
'''
specifications for dynamic system
'''
dtime=0.1 # define time step !!
iter_n = 0 # keep track of number of integration iterations !!
ti = 0 # define initial time !!
tf = 1 # define final time !!
t_steps = int((tf-ti)/dtime) # number of steps per episode
params={'a_p' : 0.5, 'b_p' : 1} #  !!
'''
define policy network and other learning/reporting parameters
'''
# define ANN with 6 hidden, 3 input, 1 output
policy = PolicyNetwork(6, 3, 1) # !!!
optimizer = optim.Adam(policy.parameters(), lr=1e-4)
All_rewards_l=[]
std_sqr = 1.0 #remember this is reduced first iter !!
episode_update_n=1 # every how many episodes I update !!
rewards_l = [None for i in range(episode_update_n)] # keep track of rewards per episode
# bvelowe
log_probs_l = [[None for j in range(t_steps)] for i in range(episode_update_n)]
epi_n=-1 # because I sum one
'''
----------------------------------------------------
Function to compute a single run given a policy -start
'''
def ComputeRun(policy_CR, initial_state_CR, plot_CR=False, t_steps_CR=t_steps):
    ''' lists for plotting '''
    if plot_CR:
        U_u_CR = [None for i in range(t_steps)]
        y1_CR = [None for i in range(t_steps)]
        y2_CR = [None for i in range(t_steps)]
        t_CR = [0 for i in range(t_steps)]
    ''' define initial conditions numpy & pytorch '''
    tj=np.array([ti])
    initial_state_I=initial_state_CR # define initial state for Integrator
    initial_state_P = np.hstack([initial_state_I,tf-tj]) # define initial state for Plicy calculation
    initial_state_P = Variable(torch.Tensor(initial_state_P)) # make it a torch variable
    # -- to run iterations (end) -- #
    for step_j in range(t_steps_CR):
        ''' compute the control by the ANN '''
        controls=policy_CR(initial_state_P)
        if plot_CR:
            action = select_action(controls[0], std_sqr, train=False)
        elif not plot_CR:
            action, log_prob_a, entropy  = select_action(controls[0], std_sqr, train=True)
        contrl={'U_u':float64(action)}
        ''' integrate the system for dtime=0.1 '''
        final_state = Model_Integrator.ModelIntegration(params,initial_state_I,contrl,dtime)
        ''' calculate probability of action taken '''
        if not plot_CR:
            log_probs_l[epi_n][step_j]=log_prob_a # global var
        initial_state_I=copy.deepcopy(final_state)
        tj=tj+dtime # calculate next time
        initial_state_P=np.hstack([initial_state_I,tf-tj])
        initial_state_P = Variable(torch.Tensor(initial_state_P)) # make it a torch variable
        ''' lists for plotting '''
        if plot_CR:
            y1_CR[step_j] = final_state[0]
            y2_CR[step_j] = final_state[1]
            t_CR[step_j] += tj
            U_u_CR[step_j] = float64(action)
    reward_CR = final_state[1]
    if plot_CR:
        return reward_CR, y1_CR, y2_CR, t_CR, U_u_CR
    if not plot_CR:
        return reward_CR
'''
Function to compute a single run given a policy -end
----------------------------------------------------
'''
''' --- Pre-trainning --- '''
inputs_PT = [i_PT*5.0/t_steps for i_PT in range(t_steps)]
runs_PT = 100
pert_size = 0.1
y1_PT, y2_PT, t_PT, U_u_PT = PreTraining(policy, inputs_PT, runs_PT,
                                 pert_size, initial_state_I=np.array([1,0]))
y1_PT_Torch = Variable(torch.Tensor(y1_PT))
y2_PT_Torch = Variable(torch.Tensor(y2_PT))
t_PT_Torch = Variable(torch.Tensor(t_PT))
''' --- Evaluate Pre-trainning --- '''

U_PT_plot = [None for i_PT in range(t_steps)]
for point in range(t_steps):
    # evaluate model with real data
    inp_point = Variable(torch.Tensor([y1_PT_Torch[point],y2_PT_Torch[point],t_PT_Torch[point]]))
    output = policy(inp_point)
    output_np = output.data.numpy()
    # compile results
    U_PT_plot[point]= output_np[0]

plt.subplot2grid((2,4),(0,0),colspan=2)
plt.plot(y1_PT)
plt.ylabel('y1 ',rotation= 360,fontsize=15)
plt.xlabel('time',fontsize=15)

plt.subplot2grid((2,4),(0,2),colspan=2)
plt.plot(y2_PT)
plt.ylabel('y2 ',rotation= 360,fontsize=15)
plt.xlabel('time',fontsize=15)

plt.subplot2grid((2,4),(1,0),colspan=2)
plt.plot(U_PT_plot)
plt.ylabel('u',rotation= 360,fontsize=15)
plt.xlabel('time',fontsize=15)

plt.subplot2grid((2,4),(1,2),colspan=2)
plt.plot(U_PT_plot)
plt.plot(U_u_PT,'ro')
plt.ylabel('u',rotation= 360,fontsize=15)
plt.xlabel('time',fontsize=15)
plt.show()

''' --- Finished Pre-trainning --- '''

'''
---------------------------------------------------
                 MAIN LOOP start
---------------------------------------------------
'''
for i_episode in range(episode_n):
    ''' --- COMPUTE EPISODE FOR LATER REINFORCEMENT LEARNING --- '''
    '''diminish std to focus more on explotation'''
    if i_episode%std_sqr_red[1]==0:
        std_sqr = std_sqr*std_sqr_red[0]
    '''run episode with network "policy" '''
    rewards_l[epi_n] = ComputeRun(policy, np.array([1,0]), plot_CR=False)
    epi_n = epi_n+1
    ''' --- TRAIN CURRENT POLICY NETWORK USING COMPUTED TRAJECTORIES --- '''
    if i_episode!=0 and i_episode%episode_update_n==0:
        gamma = 0.99
        loss = 0
        '''One reward for each episode, state includes time'''
        for i_ep in range(len(log_probs_l)):
            R = torch.zeros(1)#, 1)
            for i_j in reversed(range(len(log_probs_l[i_ep]))): # not sure why
                R = gamma * R + rewards_l[i_ep]
                loss = loss - (log_probs_l[i_ep][i_j]*(Variable(R).expand_as(log_probs_l[i_ep][i_j]))).sum()
                # A3C: We add entropy to the loss to encourage exploration: https://github.com/dennybritz/reinforcement-learning/issues/34
        loss = loss/(episode_update_n)
        #print('rewards_l = ',sum(rewards_l))
        optimizer.zero_grad()
        loss.backward()
        #utils.clip_grad_norm(policy.parameters(), 40)
        # https://discuss.pytorch.org/t/about-torch-nn-utils-clip-grad-norm/13873/2
        optimizer.step()
        # define new rewards and log probs for next episodes to train (like a "zero_grad")
        epi_n=0
        rewards_l = [None for i in range(episode_update_n)]
        log_probs_l = [[None for j in range(t_steps)] for i in range(episode_update_n)]
    ''' --- PLOT CURRENT MEAN POLICY --- '''
    if i_episode%record_n==0:
        # -- plotting lists (start) -- #
        r_report, y1_l, y2_l, t_l, U_u_l = ComputeRun(policy, np.array([1,0]), plot_CR=True)

        print('i_episode = ',i_episode,'     current_reward = ',r_report)
        All_rewards_l.append(r_report)
        print('std_sqr = ',std_sqr)
        plt.subplot2grid((2,4),(0,0),colspan=2)
        plt.plot(y1_l)
        plt.ylabel('y1 ',rotation= 360,fontsize=15)
        plt.xlabel('time',fontsize=15)

        plt.subplot2grid((2,4),(0,2),colspan=2)
        plt.plot(y2_l)
        plt.ylabel('y2 ',rotation= 360,fontsize=15)
        plt.xlabel('time',fontsize=15)

        plt.subplot2grid((2,4),(1,0),colspan=2)
        plt.plot(U_u_l)
        plt.ylabel('u',rotation= 360,fontsize=15)
        plt.xlabel('time',fontsize=15)
        plt.savefig('FiguresBatchReactor\Profile_iter_a_'+str(i_episode)+'_REINFORCE_v3.png')
        plt.close()
        torch.save(policy, 'ANNs\ANN_Prerain_10-6_v1.pt')
'''
---------------------------------------------------
                 MAIN LOOP end
---------------------------------------------------
'''
print('finished learning')
print('std_sqr = ',std_sqr)
plt.plot(All_rewards_l)
plt.ylabel('reward value')
plt.show()
