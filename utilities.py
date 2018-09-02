# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn.utils as utils
# other imports
import copy
import sys
# model integration imports
import numpy as np
import model_integrator
from model_integrator import model_integration

def normal_np(act, mu, sigma_sq):
    a = np.exp(-(act-mu)**2/(2.*sigma_sq**2))
    b = 1./np.sqrt((2.*sigma_sq**2*np.pi))
    return a*b

pi = Variable(torch.FloatTensor([np.pi]))
def normal_torch(act, mu, sigma_sq):
    a = (-1*(Variable(act)-mu).pow(2)/(2*sigma_sq**2)).exp()
    b = 1/np.sqrt((2*sigma_sq**2*pi))
    return a*b

# prints for 2 controls
def select_action(control_mean, control_sigma, train=True):
    # NOTE: not sure if this works for vectorial controls, check
    # NOTE: should return only one prob
    # control_sigma =  tensor([ 0.4420,  0.3498])
    # control_mean =  tensor([ 2.9110,  2.0363])
    if train==True: # want control only or also probabilities
        eps = torch.FloatTensor([torch.randn(control_mean.size())])
        # control_sigma =  tensor([ 0.4420,  0.3498])
        control_choice = (control_mean + np.sqrt(control_sigma)*Variable(eps)).data
        # control_choice =  tensor([ 3.2173,  2.0184])
        prob = normal_torch(control_choice, control_mean, control_sigma)
        # prob =  tensor([ 0.5396,  0.6742])
        log_prob = prob.log()
        # entropy is to explore low likelihood places
        entropy = -0.5*((control_sigma+2*pi).log()+1)
        return control_choice, log_prob, entropy
    elif train==False:
        return control_mean

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

def pretraining(policy_PT, params, dtime, tf, states_n, control_inputs, runs_PT,
 pert_size, initial_state_I=np.array([1,0])):# manually change control !!

    # define lists to compile states
    t_steps = len(control_inputs[0]) # number of steps per episode
    states_PT = [[[None for i_PT in range(states_n)]
     for i_PT in range(t_steps)]
      for i_state in range(runs_PT)]
    # define list to compile time
    t_PT = [[None for i_PT in range(t_steps)]  for i_PT in range(runs_PT)]
    # define list to compile control
    U_u_PT = [[[None for i_step in range(len(control_inputs))]
     for i_run in range(t_steps)]
      for i_control in range(runs_PT)]

    ''' # --------- computing data for later trainning -start --------- # '''

    for i_episode in range(runs_PT):
        tj=np.array([0.0]) # define initial time at each episode as 0 !!
        controls = copy.deepcopy(control_inputs)

        for step_j in range(t_steps):
            # obtain a perturbed control action for each control for each step
            for state_cntrl in range(len(controls)):
                    controls[state_cntrl][step_j] = controls[state_cntrl][step_j] * (1.
                     + np.random.uniform(-pert_size,pert_size))

            action = controls[0][step_j]
            contrl={'U_u':float(action)} # !! manually, [0] (first and only control)
            final_state = model_integration(params,initial_state_I,contrl,dtime)
            initial_state_I=copy.deepcopy(final_state)
            tj=tj+dtime # calculate next time
            # compile states
            for j_state in range(states_n):
                states_PT[i_episode][step_j][j_state] = final_state[j_state]
            # compile time
            t_PT[i_episode][step_j] = tf-tj #!! time here is tf-time
            # compile perturbed controls
            for j_control in range(len(control_inputs)):
                U_u_PT[i_episode][step_j][j_control] = controls[j_control][step_j]
    ''' # --------- computing data for later trainning -end --------- # '''

    ''' # --------- Trainning data -start --------- # '''
    # setting data for trainning
    # controls
    y_data = [[[None for k_cntrl_state in range(0,len(control_inputs))]
     for k_step in range(0,t_steps)]
      for k_episode in range(0,runs_PT)]

    for y_i in range(len(y_data)):#episode
        for y_ii in range(len(y_data[y_i])):# step
            for y_iii in range(len(y_data[y_i][y_ii])):# contrl
                y_data[y_i][y_ii][y_iii] = U_u_PT[y_i][y_ii][y_iii]# state
            y_data[y_i][y_ii]=tuple(y_data[y_i][y_ii])

    # states
    x_data = [[[None for k_state in range(0,states_n)]
     for kk_step in range(0,t_steps)]
      for kk_episode in range(0,runs_PT)]

    for x_i in range(len(x_data)):#episode
        for x_ii in range(len(x_data[x_i])):# step
            for x_iii in range(len(x_data[x_i][x_ii])):# contrl
                x_data[x_i][x_ii][x_iii] = states_PT[x_i][x_ii][x_iii]# state
            x_data[x_i][x_ii]=tuple([*x_data[x_i][x_ii],float(t_PT[x_i][x_ii])])
            # x_data[x_i][x_ii]=tuple(x_data[x_i][x_ii]) !! this one excludes time as a state

    # passing data as torh vectors
    inputs_l = [Variable(torch.Tensor(x_data[i])) for i in range(0,len(x_data))]
    labels_l = [Variable(torch.Tensor(y_data[j])) for j in range(0,len(y_data))]
    criterion = nn.MSELoss()
    optimizer = optim.Adam(policy_PT.parameters(), lr=1e-2)
#    optimizer = torch.optim.LBFGS(policy_PT.parameters(), history_size=10000)
#    def closure():
#        return PT_loss
    epoch_n = 10
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
    ''' # --------- Trainning data -end --------- # '''
    return policy_PT

def compute_run(policy_CR, params_CR, dtime_CR, tf_CR, states_n_CR, control_n_CR,
 t_steps_CR, runs_CR, std_sqr, initial_state_CR=np.array([1,0]), plot_CR=False):
    if t_steps_CR != tf_CR/dtime_CR:
        print('t_steps and tf/dtime do not match t_step = ',t_steps_CR,' tf/dtime = ',tf_CR/dtime_CR)
    ''' lists for plotting '''
    if plot_CR:
        controls_CR = [[None for i_control in range(control_n_CR)]
         for i_step in range(t_steps_CR)] # list to compile controls
        states_CR = [[None for i_state in range(states_n_CR)]
         for i_step in range(t_steps_CR)] # list to compile states
        t_CR = [0 for i in range(t_steps_CR)] # list to compile states
    else:
        log_probs_l = [None for j_step in range(t_steps_CR)]
    ''' define initial conditions numpy & pytorch '''
    tj_CR=np.array([0.])
    initial_state_I=initial_state_CR # define initial state for Integrator
    initial_state_P = np.hstack([initial_state_I,tf_CR-tj_CR]) # define initial state for Plicy calculation
    initial_state_P = Variable(torch.Tensor(initial_state_P)) # make it a torch variable
    # -- to run iterations (end) -- #
    for step_j in range(t_steps_CR):
        ''' compute the control by the ANN '''
        controls=policy_CR(initial_state_P)
        if plot_CR:
            action = select_action(controls[0], std_sqr, train=False) # also depends if vector !!
        elif not plot_CR:
            action, log_prob_a, entropy  = select_action(controls[0], std_sqr, train=True)
        contrl={'U_u':float(action)}
        #print('contrl = ',contrl)
        #print('action = ',action)
        #print('log_prob_a = ',log_prob_a)
        #print('controls = ',controls)

        # integrate the system for dtime=0.1
        final_state = model_integrator.model_integration(params_CR,initial_state_I,contrl,dtime_CR)

        # calculate probability of action taken
        if not plot_CR:
            log_probs_l[step_j]=log_prob_a # global var
            #print('log_probs_l = ',log_probs_l)
            #print('log_probs_l[step_j] = ',log_probs_l[step_j])

        initial_state_I=copy.deepcopy(final_state)
        tj_CR = tj_CR + dtime_CR # calculate next time
        initial_state_P=np.hstack([initial_state_I,tf_CR-tj_CR])
        initial_state_P = Variable(torch.Tensor(initial_state_P)) # make it a torch variable

        # lists for plotting
        if plot_CR:
            for state_k in range(states_n_CR):
                states_CR[step_j][state_k] = final_state[state_k]
            states_CR[step_j] = tuple(states_CR[step_j])
            t_CR[step_j] = tj_CR
            for control_k in range(control_n_CR):
                #controls_CR[step_j][control_k] = action[control_k] # if contrl vector
                controls_CR[step_j][control_k] = action.item() # if contrl vector
            controls_CR[step_j] = tuple(controls_CR[step_j])

    # getting rewards and finilizing
    reward_CR = final_state[1] # manual assigment of reward !!
    if plot_CR:
        if reward_CR != states_CR[-1][-1]:
            print('reward_CR seems not to match state at last step')
        return reward_CR, states_CR, t_CR, controls_CR
    if not plot_CR:
        return reward_CR, log_probs_l
